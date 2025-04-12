import pandas as pd
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add the parent directory to the path for imports
import sys
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from tensorboardX import SummaryWriter

import warnings
import time
import logging
logger = logging.getLogger(__name__)
from core.model import *
from core.train import *
from utils.checkpoint import *
from utils.util import *
from accelerate import Accelerator
from core.interp import *
from core.pygmo_fusemoe import PyGMOFuseMoE


def main():
    args = parse_args()
    print(args)

    # Check if GPU is available and determine type
    if torch.cuda.is_available():
        device_type = "cuda"
        print("CUDA GPU is available")
        if args.fp16:
            args.mixed_precision = "fp16"
        else:
            args.mixed_precision = "no"
    elif hasattr(torch, 'backends') and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device_type = "mps"
        print("Apple MPS (Metal) is available, but fp16 is not supported - disabling fp16")
        # MPS doesn't support fp16 mixed precision, so force to "no"
        args.mixed_precision = "no"
        args.fp16 = False
    else:
        device_type = "cpu"
        print("Running on CPU - disabling fp16")
        args.mixed_precision = "no"
        args.fp16 = False
    
    accelerator = Accelerator(mixed_precision=args.mixed_precision, cpu=args.cpu)

    device = accelerator.device
    print(f"Using device: {device}")
    os.makedirs(args.output_dir, exist_ok = True)
    if args.tensorboard_dir!=None:
        writer = SummaryWriter(args.tensorboard_dir)
    else:
        writer=None

    warnings.filterwarnings('ignore')
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    if args.seed is not None:
        set_seed(args.seed)

    make_save_dir(args)

    if args.seed==0:
        copy_file(args.ck_file_path+'model/', src=os.getcwd())
    if args.mode=='train':
        if 'Text' in args.modeltype:
            BioBert, BioBertConfig,tokenizer=loadBert(args,device)
        else:
            BioBert,tokenizer=None,None
        train_dataset, train_sampler, train_dataloader=data_perpare(args,'train',tokenizer)
        val_dataset, val_sampler, val_dataloader=data_perpare(args,'val',tokenizer)
        _, _, test_data_loader=data_perpare(args,'test',tokenizer)

    # Check if we should use PyGMO-enhanced FuseMOE
    use_pygmo = hasattr(args, 'use_pygmo') and args.use_pygmo
    
    if use_pygmo and args.modeltype == 'TS':
        # Use PyGMO-enhanced FuseMOE model
        print("Creating PyGMO-enhanced FuseMOE model...")
        
        # Create MoEConfig
        config = MoEConfig(
            num_experts=args.num_experts,
            moe_input_size=17,  # Hardcoded for now, should be extracted from data shape
            moe_hidden_size=args.hidden_size if hasattr(args, 'hidden_size') else 64,
            moe_output_size=args.num_labels,
            router_type=args.router_type if hasattr(args, 'router_type') else 'joint',
            gating=args.gating_function if hasattr(args, 'gating_function') else 'laplace',
            top_k=args.top_k if hasattr(args, 'top_k') else 2,
            hidden_dim=args.embed_dim
        )
        
        model = PyGMOFuseMoE(
            config=config,
            input_size=17,  # Hardcoded for now, should match moe_input_size
            hidden_size=args.hidden_size if hasattr(args, 'hidden_size') else 64,
            output_size=args.num_labels,
            num_experts=args.num_experts
        )
    elif 'Text' in args.modeltype:
        # text and ts fusion
        model= MULTCrossModel(args=args,device=device,orig_d_ts=17, orig_reg_d_ts=34, orig_d_txt=768,ts_seq_num=args.tt_max,text_seq_num=args.num_of_notes,Biobert=BioBert)
    else:
        # pure time series
        model= TSMixed(args=args,device=device,orig_d_ts=17,orig_reg_d_ts=34, ts_seq_num=args.tt_max)
    print(device)

    if args.modeltype=='TS':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.ts_learning_rate)
    elif args.modeltype=='Text' or args.modeltype=='TS_Text':
        optimizer= torch.optim.Adam([
                {'params': [p for n, p in model.named_parameters() if 'bert' not in n]},
                {'params':[p for n, p in model.named_parameters() if 'bert' in n], 'lr': args.txt_learning_rate}
            ], lr=args.ts_learning_rate)
    else:
        raise ValueError("Unknown modeltype in optimizer.")

    model, optimizer, train_dataloader,val_dataloader,test_data_loader = \
    accelerator.prepare(model, optimizer, train_dataloader,val_dataloader,test_data_loader)

    # Run PyGMO optimization if requested
    if use_pygmo and hasattr(model, 'optimize_model'):
        print("Running PyGMO optimization...")
        
        # Create a batch of training data for optimization
        # In practice, you would use more data, but for demo purposes we'll just use a small batch
        try:
            # Get a batch of data
            for batch in train_dataloader:
                if batch is not None:
                    # Extract input data and labels
                    ts_input_sequences, ts_mask_sequences, ts_tt, reg_ts_input, *_ = batch
                    
                    # Prepare data for optimization
                    train_inputs = ts_input_sequences
                    train_labels = batch[-4]  # Extract labels from the batch
                    
                    # Run optimization
                    model.optimize_model(
                        train_data=(train_inputs, train_labels),
                        expert_algo=args.expert_algorithm if hasattr(args, 'expert_algorithm') else 'de',
                        gating_algo=args.gating_algorithm if hasattr(args, 'gating_algorithm') else 'pso',
                        expert_pop_size=args.expert_population_size if hasattr(args, 'expert_population_size') else 10,
                        gating_pop_size=args.gating_population_size if hasattr(args, 'gating_population_size') else 10,
                        seed=args.seed,
                        device=device
                    )
                    break
            
            print("PyGMO optimization complete.")
        except Exception as e:
            print(f"Error during PyGMO optimization: {str(e)}")
            print("Continuing with standard training...")

    trainer_irg(model=model,args=args,accelerator=accelerator,train_dataloader=train_dataloader,\
        dev_dataloader=val_dataloader, test_data_loader=test_data_loader, device=device,\
        optimizer=optimizer,writer=writer)
    eval_test(args,model,test_data_loader, device)


if __name__ == "__main__":

    import time
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
