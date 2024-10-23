import torch 
from model import LLM, LMConfig
from train_single import train
from dataclasses import dataclass 
from typing import Optional
from data.prepare import ShakespeareCharDataset, char_tokenizer

# class NanoLlamaConfig:
#     """from https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/api/args.py"""
#     dim: int = 768
#     n_layers: int = 6
#     n_heads: int = 6
#     n_kv_heads: Optional[int] = 2
#     vocab_size: int = 65
#     multiple_of: int = 64  # make SwiGLU hidden layer size multiple of large power of 2
#     ffn_dim_multiplier: Optional[float] = 4
#     norm_eps: float = 1e-5
#     rope_theta: float = 100000
#     use_scaled_rope: bool = False

#     max_batch_size: int = 64
#     max_seq_len: int = 256
#     kv_caching: bool = False
#     pos_emb: str = 'RoPE'

#     def __init__(self, **kwargs):
#         for k, v in kwargs.items():
#             if hasattr(self, k):
#                 setattr(self, k, v)

#         if self.n_kv_heads is None:
#             self.n_kv_heads = self.n_heads
#         assert self.n_kv_heads <= self.n_heads
#         assert self.n_heads % self.n_kv_heads == 0
#         assert self.dim % self.n_heads == 0
max_seq_len = 256
max_batch_size = 64
train_dataset = ShakespeareCharDataset('data/train.txt', tokenizer=char_tokenizer(), seq_len = max_seq_len)
eval_dataset = ShakespeareCharDataset('data/eval.txt', tokenizer=char_tokenizer(), seq_len = max_seq_len)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=max_batch_size,
                        shuffle=True)#, num_workers=2)
print(eval_dataset)
eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=max_batch_size,
                        shuffle=True)#, num_workers=2)

device = 'cuda'

@dataclass
class train_config:
    learning_rate: float = 1e-3
    decay_lr: bool = True
    lr_decay_iters: int = 5000
    max_iters: int = 5000
    max_eval_iters: int = 100
    min_lr: float = 1e-4
    warmup_iters: int = 100
    num_epochs: int = 1
    eval_steps: int = 200
    out_dir: str = 'out'
    device_type: str = 'cuda'
    ptdtype: str = 'float32'
    grad_clip: float = 1.0
    gradient_accumulation_steps: int = 1
    use_profiler: bool = True
    profiler_dir: str = 'benchmark'

model = LLM(LMConfig).to(device)
print("number parameters of Llama Model %d"% model.get_num_params())

optimizer = torch.optim.AdamW(model.parameters(), lr=train_config.learning_rate)
# import wandb
# wandb_project = 'shakespeare'
# wandb_run_name = 'llama'
# wandb_run = wandb.init(project=wandb_project, name=wandb_run_name)
wandb_run = False

train(model, optimizer, train_dataloader, eval_dataloader, device, train_config, wandb_run)

