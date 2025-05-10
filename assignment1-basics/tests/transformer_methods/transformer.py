import torch
import torch.nn as nn

from .rmsnorm import RMSNorm
from .attention import MultiHeadAttention, softmax
from .swiglu import SwiGLU
from .linear import Linear
from .embedding import Embedding

import copy

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class TransformerBlock(nn.Module):
    def __init__(self, d_model:int, num_heads:int, d_ff:int, rope:nn.Module=None, dtype=torch.float32, device=None):
        super().__init__()

        self.norm1 = RMSNorm(
            d_model=d_model,
            device=device,
            dtype=dtype
        )

        self.norm2 = RMSNorm(
            d_model=d_model,
            device=device,
            dtype=dtype
        )

        self.mha = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            rope=rope,
            dtype=dtype,
            device=device
        )

        self.ff = SwiGLU(d_model=d_model, d_ff=d_ff)

    def forward(self, x:torch.Tensor):
        y = x + self.mha(self.norm1(x))
        out = y + self.ff(self.norm2(y))
        return out
    

class TransformerLM(nn.Module):
    def __init__(
            self, vocab_size:int, context_length:int, num_layers:int, 
            d_model: int, d_ff:int, num_heads:int,
            rope:nn.Module=None, device=None, dtype=None
        ):
        super().__init__()

        # self.vocab_size = vocab_size
        # self.context_length = context_length
        # self.num_layers = num_layers
        # self.device = device
        # self.dtype = dtype

        self.embedding = Embedding(num_embeddings=vocab_size, embedding_dim=d_model, device=device, dtype=dtype)
        self.transformer_blocks = clones(TransformerBlock(
            d_model=d_model, num_heads=num_heads, d_ff=d_ff, rope=rope, device=device, dtype=dtype
        ), num_layers)

        self.norm = RMSNorm(d_model=d_model, device=device, dtype=dtype)

        self.linear = Linear(d_model, vocab_size, dtype=dtype, device=device)

        self.soft = softmax

    def forward(self, x:torch.Tensor):
        h = self.embedding(x)
        for block in self.transformer_blocks:
            h = block(h)
        h = self.norm(h)
        h = self.linear(h)
        # h = self.soft(h, dim=-1)
        return h