# Copyright (c) 2024, CHAOAI INC. All rights reserved.

# Train llama model with single GPU, MPS or CPU, based on https://github.com/karpathy/nanoGPT

import math 
import torch 
import os 

def get_lr(it, config):# learning_rate, min_lr, lr_decay_iters, warmup_iters):
    # 1) linear warmup for warmup_iters steps
    if config.decay_lr:
        if it < config.warmup_iters:
            return config.learning_rate * it / config.warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > config.lr_decay_iters:
            return config.min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
        return config.min_lr + coeff * (config.learning_rate - config.min_lr)
    else:
        return config.learning_rate

def train(model, optimizer, train_dataloader, eval_dataloader, device, config, wandb_run):
    best_val_loss = 1e9
    for epoch in range(config.num_epochs):
        total_step = 0
        model.train()
        for step, batch in enumerate(train_dataloader):
            lr = get_lr(total_step, config)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            for key in batch.keys():
                batch[key] = batch[key].to(device)
            logits, loss = model(**batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_step += 1
            if wandb_run:
                wandb_run.log({
                    'train/epoch': epoch + 1,
                    'train/step': epoch * len(train_dataloader) + step,
                    'train/loss': loss.detach().float(),
                })
            print('step: %d, train/loss %.4f'%(total_step,loss.detach().float()))

            if total_step %  config.eval_steps ==0:
                model.eval()
                eval_loss = 0.0
                for step, batch in enumerate(eval_dataloader):
                    for key in batch.keys():
                        batch[key] = batch[key].to(device)
                    with torch.no_grad():
                        logits, loss = model(**batch)
                        eval_loss += loss.item()
                    if step > config.max_eval_iters:
                        break
                eval_loss /= step+1
                if wandb_run:
                    wandb_run.log({
                        'test/epoch': epoch + 1,
                        'test/step': total_step,
                        'test/loss': eval_loss,
                    })
                print('step: %d, test/loss %.4f'%(total_step,eval_loss))
                
                
                
                if eval_loss < best_val_loss or config.always_save_checkpoint:
                    if total_step > 0:
                        checkpoint = {
                            'model': model.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'epoch': epoch + 1,
                            'iter_num': total_step,
                            'best_val_loss': best_val_loss,
                            # 'config': config,
                        }
                        print(f"saving checkpoint to {config.out_dir}")
                        torch.save(checkpoint, os.path.join(config.out_dir, 'ckpt.pt'))
                
            if total_step > config.max_iters: # to be change later
                break






        

        



