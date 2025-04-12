#pylint: disable=E1101
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, LayerNorm
import numpy as np

from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
from utils.config import MoEConfig
from core.sparse_moe import MoE as MixtureOfExperts
from core.hme import HierarchicalMoE
import sys
import pdb


class Outer(nn.Module):
    def __init__(self,
                 inp1_size: int = 128,
                 inp2_size: int = 128,
                 n_neurons: int = 128):
        super(Outer, self).__init__()
        self.inp1_size = inp1_size
        self.inp2_size = inp2_size
        self.feedforward = nn.Sequential(
            nn.Linear((inp1_size + 1) * (inp2_size + 1), n_neurons),
            nn.ReLU(),
            nn.Linear(n_neurons, n_neurons),
            nn.ReLU(),
        )

    def forward(self, inp1, inp2):
        batch_size = inp1.size(0)
        append = torch.ones((batch_size, 1)).type_as(inp1)
        inp1 = torch.cat([inp1, append], dim=-1)
        inp2 = torch.cat([inp2, append], dim=-1)
        fusion = torch.zeros((batch_size, self.inp1_size + 1, self.inp2_size + 1)).type_as(inp1)
        for  i  in  range ( batch_size ):
            fusion[i] = torch.outer(inp1[i], inp2[i])
        fusion = fusion.flatten(1)

        return self.feedforward(fusion)



class MAGGate(nn.Module):
    def __init__(self, inp1_size, inp2_size, dropout):
        super(MAGGate, self).__init__()

        self.fc1 = nn.Linear(inp1_size + inp2_size, 1)
        self.fc3 = nn.Linear(inp2_size, inp1_size)
        self.beta = nn.Parameter(torch.randn((1,)))
        self.norm = nn.LayerNorm(inp1_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inp1, inp2):
        w2 = torch.sigmoid(self.fc1(torch.cat([inp1, inp2], -1)))
        adjust = self.fc3(w2 * inp2)
        one = torch.tensor(1).type_as(adjust)
        alpha = torch.min(torch.norm(inp1) / torch.norm(adjust) * self.beta, one)
        output = inp1 + alpha * adjust
        output = self.dropout(self.norm(output))
        return output


class gateMLP(nn.Module):
    def __init__(self,input_dim,hidden_size,output_dim,dropout=0.1):
        super().__init__()

        self.gate = nn.Sequential(
             nn.Dropout(dropout),
             nn.Linear(input_dim, hidden_size),
             nn.ReLU(),
             nn.Linear(hidden_size,output_dim),
             nn.Sigmoid()
        )


        self._initialize()

    def _initialize(self):
        for model in [self.gate]:
            for layer in model:
                if type(layer) in [nn.Linear]:
                    torch.nn.init.xavier_normal_(layer.weight)


    def forward(self,hidden_states ):
        gate_logits = self.gate(hidden_states)
        return gate_logits


class TimeSeriesCnnModel(nn.Module):
    def __init__(self,input_size,n_filters,filter_size,dropout,length,n_neurons,layers):
        super().__init__()

        padding = int(np.floor(filter_size / 2))
        self.layers=layers
        if layers>=1:
            self.conv1 = nn.Conv1d(input_size, n_filters, filter_size, padding=padding)
            self.pool1 = nn.MaxPool1d(2, 2)

        if layers>=2:
            self.conv2 = nn.Conv1d(n_filters, n_filters, filter_size, padding=padding)
            self.pool2 = nn.MaxPool1d(2, 2)

        if layers>=3:
            self.conv3 = nn.Conv1d(n_filters, n_filters, filter_size, padding=padding)
            self.pool3 = nn.MaxPool1d(2, 2)

        self.fc1 = nn.Linear(int(length * n_filters / (2**layers)), n_neurons)
        self.fc1_drop = nn.Dropout(dropout)

    def forward(self, x):
        if self.layers>=1:
            x = self.pool1(F.relu(self.conv1(x)))
        if self.layers>=2:
            x = self.pool2(F.relu(self.conv2(x)))
        if self.layers>=3:
            x = self.pool3(F.relu(self.conv3(x)))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1_drop(self.fc1(x)))

        return x

# F.gumbel_softmax(logits, tau=1, hard=True)

class multiTimeAttention(nn.Module):
    "mTAND module"
    def __init__(self, input_dim, nhidden=16,
                 embed_time=16, num_heads=1):
        super(multiTimeAttention, self).__init__()
        assert embed_time % num_heads == 0
        self.embed_time = embed_time
        self.embed_time_k = embed_time // num_heads
        self.h = num_heads
        self.dim = input_dim
        self.nhidden = nhidden
        self.linears = nn.ModuleList([nn.Linear(embed_time, embed_time),
                                      nn.Linear(embed_time, embed_time),
                                      nn.Linear(input_dim*num_heads, nhidden)])

    def attention(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"

        dim = value.size(-1)
        d_k = query.size(-1)
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(d_k)
        scores = scores.unsqueeze(-1).repeat_interleave(dim, dim=-1)
        if mask is not None:
            if len(mask.shape)==3:
                mask=mask.unsqueeze(-1)

            scores = scores.masked_fill(mask.unsqueeze(-3) == 0, -10000)
        p_attn = F.softmax(scores, dim = -2)
        if dropout is not None:
            p_attn=F.dropout(p_attn, p=dropout, training=self.training)
        return torch.sum(p_attn*value.unsqueeze(-3), -2), p_attn

    def forward(self, query, key, value, mask=None, dropout=0.1):
        "Compute 'Scaled Dot Product Attention'"
        batch, seq_len, dim = value.size()
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        value = value.unsqueeze(1)
        query, key = [l(x).view(x.size(0), -1, self.h, self.embed_time_k).transpose(1, 2)
                      for l, x in zip(self.linears, (query, key))]
        x, _ = self.attention(query, key, value, mask, dropout)
        x = x.transpose(1, 2).contiguous() \
             .view(batch, -1, self.h * dim)
        return self.linears[-1](x)


class MultiheadAttention(nn.Module):
    """Multi-headed attention.
    See "Attention Is All You Need" for more details.
    """

    def __init__(self, embed_dim, num_heads, attn_dropout=0.,
                 bias=True, add_bias_kv=False, add_zero_attn=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.in_proj_weight = Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        self.register_parameter('in_proj_bias', None)
        if bias:
            self.in_proj_bias = Parameter(torch.Tensor(3 * embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(self, query, key, value, attn_mask=None):
        """Input shape: Time x Batch x Channel
        Self-attention can be implemented by passing in the same arguments for
        query, key and value. Timesteps can be masked by supplying a T x T mask in the
        `attn_mask` argument. Padding elements can be excluded from
        the key by passing a binary ByteTensor (`key_padding_mask`) with shape:
        batch x src_len, where padding elements are indicated by 1s.
        """

        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
        kv_same = key.data_ptr() == value.data_ptr()

        # embed_dim can be decomposed into num_heads x head_dim?
        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        assert key.size() == value.size()

        aved_state = None
        if qkv_same:
            # self-attention
            q, k, v = self.in_proj_qkv(query)
        elif kv_same:
            # encoder-decoder attention
            q = self.in_proj_q(query)

            if key is None:
                assert value is None
                k = v = None
            else:
                k, v = self.in_proj_kv(key)
        else:
            q = self.in_proj_q(query)
            k = self.in_proj_k(key)
            v = self.in_proj_v(value)
        q = q * self.scaling

        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        # src_len = bsz * num_heads
        src_len = k.size(1)
        if self.add_zero_attn:
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_mask is not None:
            try:
                attn_weights += attn_mask.unsqueeze(0)
            except:
                print(attn_weights.shape)
                print(attn_mask.unsqueeze(0).shape)
                assert False

        attn_weights = F.softmax(attn_weights.float(), dim=-1).type_as(attn_weights)
        # attn_weights = F.relu(attn_weights)
        # attn_weights = attn_weights / torch.max(attn_weights)
        attn_weights = F.dropout(attn_weights, p=self.attn_dropout, training=self.training)

        attn = torch.bmm(attn_weights, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]

        # embed_dim = self.num_heads * self.head_dim
        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        # average attention weights over heads
        attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        attn_weights = attn_weights.sum(dim=1) / self.num_heads

        return attn, attn_weights

    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_kv(self, key):
        return self._in_proj(key, start=self.embed_dim).chunk(2, dim=-1)

    def in_proj_q(self, query, **kwargs):
        return self._in_proj(query, end=self.embed_dim, **kwargs)

    def in_proj_k(self, key):
        return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)

    def in_proj_v(self, value):
        return self._in_proj(value, start=2 * self.embed_dim)

    def _in_proj(self, input, start=0, end=None, **kwargs):
        weight = kwargs.get('weight', self.in_proj_weight)
        bias = kwargs.get('bias', self.in_proj_bias)
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)


class TransformerEncoder(nn.Module):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.
    Args:
        embed_tokens (torch.nn.Embedding): input embedding
        num_heads (int): number of heads
        layers (int): number of layers
        attn_dropout (float): dropout applied on the attention weights
        relu_dropout (float): dropout applied on the first layer of the residual block
        res_dropout (float): dropout applied on the residual block
        attn_mask (bool): whether to apply mask on the attention weights
    """

    def __init__(self, embed_dim, num_heads, layers, device,attn_dropout=0.0, relu_dropout=0.0, res_dropout=0.0,
                 embed_dropout=0.0, attn_mask=False,learn_embed=True, q_seq_len=None, kv_seq_len=None,):
        super().__init__()
        self.dropout = embed_dropout      # Embedding dropout
        self.attn_dropout = attn_dropout
        self.embed_dim = embed_dim
        self.embed_scale = math.sqrt(embed_dim)
        self.device=device
        self.q_seq_len=q_seq_len
        self.kv_seq_len=kv_seq_len
        if learn_embed:
            if self.q_seq_len!=None:
                self.embed_positions_q=nn.Embedding(self.q_seq_len,embed_dim,padding_idx=0)
                nn.init.normal_(self.embed_positions_q.weight, std=0.02)

            if self.kv_seq_len!=None:
                self.embed_positions_kv=nn.Embedding(self.kv_seq_len,embed_dim)
                nn.init.normal_(self.embed_positions_kv.weight, std=0.02)

        else:
            self.embed_positions = SinusoidalPositionalEmbedding(embed_dim)

        self.attn_mask = attn_mask

        self.layers = nn.ModuleList([])
        for layer in range(layers):
            new_layer = TransformerEncoderLayer(embed_dim,
                                                num_heads=num_heads,
                                                attn_dropout=attn_dropout,
                                                relu_dropout=relu_dropout,
                                                res_dropout=res_dropout,
                                                attn_mask=attn_mask)
            self.layers.append(new_layer)

        self.normalize = True
        if self.normalize:
            self.layer_norm = LayerNorm(embed_dim)

    def forward(self, x_in, x_in_k = None, x_in_v = None):
        """
        Args:
            x_in (FloatTensor): embedded input of shape `(src_len, batch, embed_dim)`
            x_in_k (FloatTensor): embedded input of shape `(src_len, batch, embed_dim)`
            x_in_v (FloatTensor): embedded input of shape `(src_len, batch, embed_dim)`
        Returns:
            dict:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
        """

        x=x_in
        length_x = x.size(0) # (length,Batch_size,input_dim)
        x = self.embed_scale * x_in
        if self.q_seq_len is not None:
            position_x = torch.tensor(torch.arange(length_x),dtype=torch.long).to(self.device)
            x += (self.embed_positions_q(position_x).unsqueeze(0)).transpose(0,1)  # Add positional embedding
        x =F.dropout(x, p=self.dropout, training=self.training)

        if x_in_k is not None and x_in_v is not None:
            # embed tokens and positions

            length_kv = x_in_k.size(0) # (Batch_size,length,input_dim)
            position_kv = torch.tensor(torch.arange(length_kv),dtype=torch.long).to(self.device)

            x_k = self.embed_scale * x_in_k
            x_v = self.embed_scale * x_in_v
            if self.kv_seq_len is not None:
                x_k += (self.embed_positions_kv(position_kv).unsqueeze(0)).transpose(0,1)   # Add positional embedding
                x_v += (self.embed_positions_kv(position_kv).unsqueeze(0)).transpose(0,1)   # Add positional embedding
            x_k = F.dropout(x_k, p=self.dropout, training=self.training)
            x_v = F.dropout(x_v, p=self.dropout, training=self.training)

        # encoder layers
        intermediates = [x]
        for layer in self.layers:
            if x_in_k is not None and x_in_v is not None:
                x = layer(x, x_k, x_v)
            else:
                x = layer(x)
            intermediates.append(x)

        if self.normalize:
            x = self.layer_norm(x)

        return x

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions())


class TransformerCrossEncoder(nn.Module):
    """
    Transformer encoder consisting of *layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.
    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        layers: Number of layers
        attn_dropout: Dropout applied on the attention weights
        relu_dropout: Dropout applied on the first layer of the residual block
        res_dropout: Dropout applied on the residual block
        attn_mask: Boolean indicating whether to apply mask on the attention weights
    """

    def __init__(self, args, embed_dim, num_heads, layers, device, attn_dropout=0.0, relu_dropout=0.0, res_dropout=0.0,
                 embed_dropout=0.0, attn_mask=False, q_seq_len_1=None, q_seq_len_2=None, num_modalities=2):
        super().__init__()
        self.device = device
        self.dropout = embed_dropout
        self.attn_dropout = attn_dropout
        self.embed_dim = embed_dim
        self.embed_scale = math.sqrt(embed_dim)
        self.embed_positions_q = nn.ModuleList([])
        self.q_seq_len_1 = q_seq_len_1
        self.q_seq_len_2 = q_seq_len_2
        self.num_modalities = num_modalities
        if q_seq_len_1 is not None:
            self.embed_positions_q = nn.ModuleList([nn.Embedding(q_seq_len_1, embed_dim) for _ in range(num_modalities)])

        self.attn_mask = attn_mask

        self.layers = nn.ModuleList([])
        for layer in range(layers):
            # Only pass args to TransformerCrossEncoderLayer, not the additional parameters
            new_layer = TransformerCrossEncoderLayer(args,
                                                    mask_self_attn=attn_mask,
                                                    mask_cross_attn=attn_mask,
                                                    device=self.device)
            self.layers.append(new_layer)

        self.normalize = True
        if self.normalize:
            self.layer_norm = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(num_modalities)])

    def forward(self, x_in_list, modality):
        """
        Args:
            x_in_list (list of FloatTensor): embedded input of shape `(src_len, batch, embed_dim)`
        Returns:
            dict:
                - **encoder_out** (Tensor): the list of last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`

        """

        # x_in_list contains ts and clinical notes tensors
        x_list = x_in_list
        lengths, positions = [], []
        for i in range(self.num_modalities):
            lengths.append(x_list[i].size(0))
        x_list = [self.embed_scale * x_in for x_in in x_in_list]
        if self.q_seq_len_1 is not None:
            for length in lengths:
                positions.append(torch.tensor(torch.arange(length),dtype=torch.long).to(self.device))
            x_list = [l(position_x).unsqueeze(0).transpose(0,1) + x for l, x, position_x in zip(self.embed_positions_q, x_list, positions)]
              # Add positional embedding
            x_list = [F.dropout(x, p=self.dropout, training=self.training) for x in x_list]
        # encoder layers
        for layer in self.layers:
            x_list = layer(x_list, modality) #proj_x_txt, proj_x_ts
            if x_list is None:
                return None

        if self.normalize:
            x_list=[l(x) for l, x in zip(self.layer_norm, x_list)]
        return x_list


class TransformerCrossEncoderLayer(nn.Module):
    def __init__(self, args, 
                 mask_self_attn=False,
                 mask_cross_attn=False,
                 device='cpu'):
        super(TransformerCrossEncoderLayer, self).__init__()

        # Store device parameter
        self.device = device
        
        # Ensure dimensions are consistent
        assert args.embed_dim == args.embed_dim

        if args.cross_method == "moe":
            # Create MoEConfig for MixtureOfExperts with all required parameters
            num_experts = args.num_of_experts[0] if args.num_of_experts else 3
            moe_input_size = args.embed_dim
            moe_hidden_size = args.hidden_size
            moe_output_size = args.embed_dim
            router_type = args.router_type if hasattr(args, 'router_type') else 'joint'
            
            moe_config = MoEConfig(
                num_experts=num_experts,
                moe_input_size=moe_input_size,
                moe_hidden_size=moe_hidden_size,
                moe_output_size=moe_output_size,
                router_type=router_type,
                top_k=args.top_k[0] if args.top_k else 2,
                num_modalities=args.num_modalities if hasattr(args, 'num_modalities') else 2,
                disjoint_top_k=args.disjoint_top_k if hasattr(args, 'disjoint_top_k') else 2,
                gating='softmax',  # Default gating method
                hidden_dim=args.embed_dim
            )
            
            # Initialize MoE with config
            self.moe = MixtureOfExperts(moe_config)
            
            # Move to the appropriate device
            self.moe = self.moe.to(self.device)
        else:
            # Initialize cross attention otherwise
            self.cross_attn = nn.MultiheadAttention(
                args.embed_dim, 
                args.num_heads, 
                dropout=args.dropout
            )
        
        self.method = args.cross_method
        self.mask_cross_attn = mask_cross_attn

        # Self attention always used
        self.self_attn = nn.MultiheadAttention(
            args.embed_dim, 
            args.num_heads, 
            dropout=args.dropout
        )
        
        self.mask_self_attn = mask_self_attn
        
        # Feed forward network
        self.linear1 = nn.Linear(args.embed_dim, args.hidden_size)
        self.dropout = nn.Dropout(args.dropout)
        self.linear2 = nn.Linear(args.hidden_size, args.embed_dim)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(args.embed_dim)
        self.norm2 = nn.LayerNorm(args.embed_dim)
        self.norm3 = nn.LayerNorm(args.embed_dim)
        self.dropout1 = nn.Dropout(args.dropout)
        self.dropout2 = nn.Dropout(args.dropout)
        self.dropout3 = nn.Dropout(args.dropout)
        
        # MoE or self_cross flag
        if args.self_cross:
            # Create MoEConfig for self_moe with all required parameters
            self_num_experts = args.num_of_experts[0] if hasattr(args, 'num_of_experts') and args.num_of_experts else 8
            self_moe_config = MoEConfig(
                num_experts=self_num_experts,
                moe_input_size=args.embed_dim,
                moe_hidden_size=args.hidden_size,
                moe_output_size=args.embed_dim,
                router_type=args.router_type if hasattr(args, 'router_type') else 'joint',
                top_k=args.top_k[0] if hasattr(args, 'top_k') and args.top_k else 4,
                num_modalities=args.num_modalities if hasattr(args, 'num_modalities') else 2,
                disjoint_top_k=args.disjoint_top_k if hasattr(args, 'disjoint_top_k') else 2,
                gating='softmax',  # Default gating method
                hidden_dim=args.embed_dim
            )
            
            # Initialize MoE with config
            self.self_moe = MixtureOfExperts(self_moe_config)
            
            # Move to the appropriate device
            self.self_moe = self.self_moe.to(self.device)
            
        self.self_cross = args.self_cross if hasattr(args, 'self_cross') else False
        
    def forward(self, q, k=None, v=None, attn_mask=None):
        """
        Args:
            q: query tensor
            k: key tensor (optional)
            v: value tensor (optional)
            attn_mask: attention mask (optional)
        Returns:
            output tensor
        """
        # Self attention with residual connection
        residual = q
        q = self.norm1(q)
        
        # Apply self attention
        if self.mask_self_attn:
            mask = buffered_future_mask(q)
        else:
            mask = None
            
        # Self attention
        q2, _ = self.self_attn(q, q, q, attn_mask=mask)
        q = residual + self.dropout1(q2)
        
        # Cross attention or MoE
        if k is not None and v is not None:
            residual = q
            q = self.norm2(q)
            
            if self.method == "moe":
                # Use MoE layer
                q2 = self.moe(q)
            else:
                # Cross attention
                if self.mask_cross_attn:
                    mask = buffered_future_mask(q, k)
                else:
                    mask = None
                q2, _ = self.cross_attn(q, k, v, attn_mask=mask)
                
            q = residual + self.dropout2(q2)
            
        # Feed forward network
        residual = q
        q = self.norm3(q)
        q2 = self.linear2(self.dropout(F.relu(self.linear1(q))))
        q = residual + self.dropout3(q2)
        
        return q


class TransformerEncoderLayer(nn.Module):
    """
    Implements a Transformer Encoder Layer used in BERT/XLM style pre-trained
    models.
    """

    def __init__(self, embed_dim, num_heads=4, attn_dropout=0.1, relu_dropout=0.1, res_dropout=0.1,
                 attn_mask=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        self.self_attn = MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            attn_dropout=attn_dropout
        )
        self.dropout1 = nn.Dropout(res_dropout)
        self.dropout2 = nn.Dropout(res_dropout)
        self.dropout3 = nn.Dropout(res_dropout)
        self.dropout = nn.Dropout(relu_dropout)
        self.res_dropout = res_dropout
        self.normalize_before = True
        self.mask_self_attn = attn_mask
        self.mask_cross_attn = attn_mask

        self.fc1 = Linear(self.embed_dim, 4*self.embed_dim)   # The "Add & Norm" part in the paper
        self.fc2 = Linear(4*self.embed_dim, self.embed_dim)
        self.layer_norms = nn.ModuleList([LayerNorm(self.embed_dim) for _ in range(2)])

    def forward(self, x, x_k=None, x_v=None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.
            x_k (Tensor): same as x
            x_v (Tensor): same as x
        Returns:
            encoded output of shape `(batch, src_len, embed_dim)`
        """

        residual = x
        x = self.layer_norms[0](x)
        
        # Apply self attention
        if self.mask_self_attn:
            mask = buffered_future_mask(x)
        else:
            mask = None
            
        # Self attention
        x2, _ = self.self_attn(x, x, x, attn_mask=mask)
        x = residual + self.dropout1(x2)
        
        # Cross attention or MoE
        if x_k is not None and x_v is not None:
            residual = x
            x = self.layer_norms[1](x)
            
            if self.method == "moe":
                # Use MoE layer
                x2 = self.moe(x)
            else:
                # Cross attention
                if self.mask_cross_attn:
                    mask = buffered_future_mask(x, x_k)
                else:
                    mask = None
                x2, _ = self.cross_attn(x, x_k, x_v, attn_mask=mask)
                
            x = residual + self.dropout2(x2)
            
        # Feed forward network
        residual = x
        x = self.fc2(self.dropout(F.relu(self.fc1(x))))
        x = residual + self.dropout3(x)
        
        return x