import torch 
import torch.nn as nn
import torch.nn.functional as F
import math 
import matplotlib.pyplot as plt 

from typing import Optional, Callable
from dataclasses import dataclass

import fire
torch.manual_seed(4567)
max_len = 256

class PositionalEmbedding(nn.Module):
    def __init__(self, dim, max_len, base=500000.0):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(- math.log(base) * torch.arange(0, dim, 2) / dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)
    def forward(self, T):
        return self.pe[:, :T]

class RoPE(nn.Module):
    def __init__(self, dim, max_len, base=10.0):
        super().__init__()
        self.pos_emb = PositionalEmbedding(dim, max_len, base)
    def forward(self, x, pos):
        sinusoidal_pos = self.pos_emb(pos).to(x.device)
        sin = sinusoidal_pos[0,:,0::2]
        cos = sinusoidal_pos[0,:,1::2]
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1]//2:]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

class Attention(nn.Module):

    def __init__(self, dim, num_heads):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.n_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)
        self.rope = RoPE(dim, max_len)


    def forward(self, x, mask: Optional[torch.Tensor]):
        B, T, C = x.shape
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        q = self.rope(q, T)
        k = self.rope(k, T)
        q = q.view(B, T, self.n_heads, self.head_dim)
        k = k.view(B, T, self.n_heads, self.head_dim)
        v = v.view(B, T, self.n_heads, self.head_dim)





        q = q.transpose(1,2) # B, n_head, T, head_dim
        k = k.transpose(1,2)
        v = v.transpose(1,2)

        scores = q @ k.transpose(-1,-2) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        scores = F.softmax(scores, dim=-1)
        output = scores @ v

        output = output.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(output)

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, activation = F.silu):
        super().__init__()
        self.linear_fc = nn.Linear(input_dim, hidden_dim, bias=False)
        self.linear_gate = nn.Linear(input_dim, hidden_dim, bias=False)
        self.linear_proj = nn.Linear(hidden_dim, output_dim ,bias=False)
        self.activation = activation
        self.hidden_dim = hidden_dim
    def forward(self, x):
        gate = self.linear_gate(x)
        logits = self.linear_proj(self.linear_fc(x) * self.activation(gate))#/math.sqrt(self.hidden_dim))
        return logits
    
class TransformerBlock(nn.Module):
    def __init__(self, dim, out_dim, num_heads, mlp_hidden_dim, mlp_act_fn):
        super().__init__()
        self.self_attn = Attention(dim, num_heads)
        self.mlp = MLP(dim, mlp_hidden_dim, out_dim, mlp_act_fn)
        self.skip = nn.Linear(dim, out_dim)
        # self.n_layers = config.n_layers
    def forward(self, x, mask: Optional[torch.Tensor]):
        # h = self.self_attn(x, mask)
        # out =  self.mlp(h) #/ math.sqrt(self.n_layers) + x
        h = x + self.self_attn(x, mask) #/ math.sqrt(self.n_layers)
        out = self.skip( h) + self.mlp(h) #/ math.sqrt(self.n_layers)
        return out
    
class LLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.vocab_size  = config.vocab_size
        self.n_layers = config.n_layers

        self.embed_tokens = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList()
        dims = [384, 192, 96, 192, 384]
        # dims = [384, 384, 384, 384, 384]
        for layer_id in range(self.n_layers):
            self.layers.append(TransformerBlock(dims[layer_id],dims[layer_id+1],2,dims[layer_id],F.silu))
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.embed_tokens.weight = self.lm_head.weight
    def forward(self, input_ids, labels = None):
        B, T = input_ids.size()
        h = self.embed_tokens(input_ids)
        mask = torch.full(
            (T, T), float("-inf"), device=input_ids.device
        )

        mask = torch.triu(mask, diagonal=1)

        for layer in self.layers:
            h =  layer(h, mask=mask)

        if labels is not None:
            logits = self.lm_head(h)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-1)
        else:
            logits = self.lm_head(h)#self.lm_head(h[:, [-1], :])
            loss = None
        return logits, loss
    @torch.no_grad()
    def generate(self, input_ids):
        logits, _ = self(input_ids)
        probs = F.softmax(logits[:,-1,:], dim=-1)
        return probs
    def save(self, path):
        return torch.save(self.state_dict(), path)
    def load(self, path):
        self.load_state_dict(torch.load(path, weights_only=True))
        return None
    def get_num_params(self):
        return sum([p.numel() for p in self.parameters()])

### 6 layers: 384 -> 96 -> 24 -> 96 -> 384 
@dataclass
class LMConfig:
    dim: int = 384
    n_layers: int = 4
    vocab_size: int = 65
    num_heads: int = 1
    mlp_hidden_dim: int = 128
    mlp_act_fn: Callable[[torch.Tensor], torch.Tensor] =  F.silu

  
# @dataclass
# class LMConfig:
#     dim: int = 384
#     n_layers: int = 4
#     vocab_size: int = 65
#     num_heads: int = 6
#     mlp_hidden_dim: int = 1024
#     mlp_act_fn: Callable[[torch.Tensor], torch.Tensor] =  F.silu

# @dataclass
# class LMConfig:
#     dim: int = 256
#     n_layers: int = 2
#     vocab_size: int = 65
#     num_heads: int = 4
#     mlp_hidden_dim: int = 256
#     mlp_act_fn: Callable[[torch.Tensor], torch.Tensor] = F.silu
