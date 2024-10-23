# Copyright (c) 2024, CHAOAI INC. All rights reserved.

"""Code based on nanoGPT: https://github.com/karpathy/nanoGPT"""
import os 
import torch
import requests


input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')

if not os.path.exists(input_file_path):
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open(input_file_path, 'w') as f:
        f.write(requests.get(data_url).text)
with open(input_file_path, 'r') as f:
    data = f.read()
print(f"length of dataset in characters: {len(data):,}")

train_file_path = os.path.join(os.path.dirname(__file__), 'train.txt')
eval_file_path = os.path.join(os.path.dirname(__file__), 'eval.txt')

with open(train_file_path, 'w') as f:
    f.write(data[:int(len(data)*0.9)])
with open(eval_file_path, 'w') as f:
    f.write(data[int(len(data)*0.9):])

chars = sorted(list(set(data)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

class char_tokenizer:
    def __init__(self):
        pass
    def encode(self, x):
        return [stoi[c] for c in x]
    def decode(self, x):
        return ''.join([itos[i] for i in x]) 
    
class ShakespeareCharDataset(torch.utils.data.Dataset):
    def __init__(self, text_file, tokenizer, seq_len):
        with open(text_file, 'r') as f:
            data = f.read()
        self.text = data
        self.input_ids = torch.LongTensor(tokenizer.encode(data))
        self.seq_len = seq_len
        
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        ix = torch.randint(len(self.input_ids) - self.seq_len,(1,))
        x = self.input_ids[ix:ix + self.seq_len]
        y = self.input_ids[ix + 1:ix + self.seq_len + 1]
        return {'input_ids':x, 'labels':y}
    
    @classmethod
    def random_split(cls):
        return 
    

# ds = ShakespeareCharDataset('input.txt', tokenizer=char_tokenizer(), seq_len = 10)
# ds_loader = torch.utils.data.DataLoader(ds, batch_size=4,
                        # shuffle=True, num_workers=0)
