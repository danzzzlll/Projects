from collections.abc import Callable
from typing import Optional
import torch
import math

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                grad = p.grad.data

                if len(state) == 0:
                    state['t'] = 0
                    state['m'] = torch.zeros_like(p.data)
                    state['v'] = torch.zeros_like(p.data)

                state['t'] += 1 
                m = beta1 * state['m'] + (1 - beta1) * grad
                v = beta2 * state['v'] + (1 - beta2) * grad * grad
                m_hat = m / (1 - beta1 ** state['t'])
                v_hat = v / (1 - beta2 ** state['t'])
                p.data -= lr * m_hat / (torch.sqrt(v_hat) + eps)
                p.data -= lr * weight_decay * p.data 
                state['m'] = m 
                state['v'] = v

        return loss
    

class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)
    def step(self, closure: Optional[Callable] = None):
        print(self.param_groups)
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p] 
                t = state.get("t", 0) 
                grad = p.grad.data 
                p.data -= lr / math.sqrt(t + 1) * grad 
                state["t"] = t + 1 
        return loss
    

class LION(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.999), weight_decay=0.0):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {
            "lr": lr,
            "betas": betas,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            weight_decay = group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                grad = p.grad.data

                if len(state) == 0:
                    state['m'] = torch.zeros_like(p.data)
                
                c = beta1 * state['m'] + (1 - beta1) * grad
                p.data -= lr * (torch.sign(c) + weight_decay * p.data)
                m = beta2 * state['m'] + (1 - beta2) * grad
                state['m'] = m
        return loss


class RMSPROP(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, gamma=0.9, eps=1e-8):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {
            "lr": lr,
            "gamma": gamma,
            "eps": eps,
        }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group['lr']
            gamma = group['gamma']
            eps = group['eps']
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                grad = p.grad.data

                if len(state) == 0:
                    state['v'] = torch.zeros_like(p.data)

                v = gamma * state['v'] + (1 - gamma) * grad * grad
                state['v'] = v
                p.data -= lr * grad / (torch.sqrt(v) + eps)
        return loss