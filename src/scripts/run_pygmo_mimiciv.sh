#!/bin/bash
# Enhanced MIMIC-IV script with PyGMO integration

export CUDA_VISIBLE_DEVICES=1

# Create output directory
OUTPUT_DIR="../run/PyGMO_TS_CXR_Text"
mkdir -p "$OUTPUT_DIR"
echo "Output directory: $OUTPUT_DIR"

python -W ignore main_mimiciv.py \
                --num_train_epochs 8 \
                --modeltype 'TS_CXR_Text' \
                --kernel_size 1 \
                --train_batch_size 1 \
                --eval_batch_size 8 \
                --seed 42 \
                --gradient_accumulation_steps 16 \
                --num_update_bert_epochs 2 \
                --bertcount 0 \
                --ts_learning_rate 0.0004 \
                --txt_learning_rate 0.00002 \
                --notes_order 'Last' \
                --num_of_notes 5 \
                --max_length 1024 \
                --layers 3 \
                --output_dir "$OUTPUT_DIR" \
                --embed_dim 128 \
                --num_modalities 3 \
                --model_name "bioLongformer" \
                --task 'ihm-48-cxr-notes-ecg' \
                --file_path '../Data/ihm' \
                --num_labels 2 \
                --num_heads 8 \
                --embed_time 64 \
                --tt_max 48 \
                --TS_mixup \
                --mixup_level 'batch' \
                --fp16 \
                --irregular_learn_emb_text \
                --irregular_learn_emb_ts \
                --irregular_learn_emb_cxr \
                --irregular_learn_emb_ecg \
                --cross_method "moe" \
                --gating_function "laplace" \
                --num_of_experts 3 5 \
                --top_k 2 4 \
                --disjoint_top_k 2 \
                --hidden_size 512 \
                --use_pt_text_embeddings \
                --router_type 'joint' \
                --reg_ts \
                --use_pygmo \
                --expert_algorithm "de" \
                --gating_algorithm "pso" \
                --expert_population_size 10 \
                --gating_population_size 10 \
                --expert_generations 5 \
                --gating_generations 5

echo "PyGMO-enhanced MIMIC-IV run completed. Results saved to $OUTPUT_DIR" 