import torch
from einops import rearrange
import math
import numpy as np
import typing
import os
import torch.nn as nn


def perplexity(losses: torch.Tensor):
    return torch.exp(torch.mean(losses, dim=-1))


def cross_entropy(logits:torch.Tensor, targets:torch.Tensor):
    max_logits = torch.max(logits, dim=-1, keepdim=True).values
    logits_stable = logits - max_logits
    log_sum_exp = torch.log(torch.sum(torch.exp(logits_stable), dim=-1))
    targets = rearrange(targets, "bs ... -> bs ... 1")
    target_logits = rearrange(logits_stable.gather(-1, targets), "bs ... 1 -> bs ...")
    losses = -target_logits + log_sum_exp
    mean_loss = torch.mean(losses)
    return mean_loss


def cosine_scheduler(t, alpha_min, alpha_max, T_w, T_c):
    if t < T_w:
        return (t / T_w) * alpha_max
    elif t >= T_w and t <= T_c:
        return alpha_min + (1/2) * (1 + math.cos((t - T_w) / (T_c - T_w) * math.pi)) * (alpha_max - alpha_min)
    else:
        return alpha_min
    

def gradient_clipping(parameters, M, eps=1e-6):
    """
    Parameters:
    - parameters: Iterable[torch.nn.Parameter] - список параметров модели
    - M: float - максимальное значение нормы
    - eps: float - малая константа для численной стабильности
    """
    grads = [p.grad for p in parameters if p.grad is not None]
    
    if not grads:
        return
    
    total_norm = torch.norm(torch.stack([torch.norm(g.detach(), 2) for g in grads]), 2)
    clip_coef = M / (total_norm + eps)
    
    if clip_coef < 1:
        for grad in grads:
            grad.mul_(clip_coef)


def batchify(x, batch_size, context_length, device=None):
    assert len(x) >= context_length + 1, "Context length should be more than len x"
    start_indices = np.random.randint(0, len(x) - context_length, size=batch_size)
    inputs = torch.tensor([x[i: i + context_length] for i in start_indices], dtype=torch.long, device=device)
    targets = torch.tensor([x[i + 1: i + context_length + 1] for i in start_indices], dtype=torch.long, device=device)
    return (inputs, targets)


def save_checkpoint(model:nn.Module, optimizer:torch.optim.Optimizer, iteration:int, out:str|os.PathLike|typing.BinaryIO|typing.IO[bytes]):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration
    }
    torch.save(checkpoint, out)


def load_checkpoint(src:str|os.PathLike|typing.BinaryIO|typing.IO[bytes], model:nn.Module, optimizer:torch.optim.Optimizer):
    checkpoint = torch.load(src)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['iteration']
