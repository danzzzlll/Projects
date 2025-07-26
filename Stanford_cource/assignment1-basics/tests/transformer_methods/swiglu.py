import torch
import torch.nn as nn
from .linear import Linear

def swish(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()

        self.w1 = Linear(d_model, d_ff)  # [d_model -> d_ff]
        self.w2 = Linear(d_ff, d_model)  # [d_ff -> d_model]
        self.w3 = Linear(d_model, d_ff)


    def forward(self, x: torch.Tensor):
        return self.w2(swish(self.w1(x)) * self.w3(x))


