import torch
import torch.nn as nn
from einops import rearrange, einsum
from .linear import Linear
import math

def softmax(x: torch.Tensor, dim: int):
    x_exp = torch.exp(x - torch.max(x, dim=dim, keepdim=True).values)
    return x_exp / torch.sum(x_exp, dim=dim, keepdim=True)


def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask=None):
    d_k = Q.size(-1)

    # K_t = rearrange(K, "batch_size ... seq_len d_k -> batch_size ... d_k seq_len")

    attn_weights = einsum(
        Q, K,
        "batch_size ... seq_len_q d_k, batch_size ... seq_len_k d_k  -> batch_size ... seq_len_q seq_len_k"
    )

    scores = attn_weights / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
        # scores.masked_fill(mask, float('-inf'))

    soft_scores = softmax(scores, dim=-1)

    attn_scores = einsum(
        soft_scores, V,
        "batch_size ... seq_len_q seq_len_k, batch_size ... seq_len_k d_v -> batch_size ... seq_len_q d_v"
    )

    return attn_scores


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, rope:nn.Module=None, device=None, dtype=None):
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.head_dim = d_model // num_heads
        self.num_heads = num_heads
        self.d_model = d_model
        self.device = device
        
        self.W_q = Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_k = Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_v = Linear(d_model, d_model, device=device, dtype=dtype)
        self.W_o = Linear(d_model, d_model, device=device, dtype=dtype)

        self.rope = rope

    def forward(self, x:torch.Tensor, token_positions:torch.Tensor = None):
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        Q_a = rearrange(Q, "batch_size seq_len (num_heads head_dim) -> batch_size num_heads seq_len head_dim", num_heads=self.num_heads, head_dim=self.head_dim)
        K_a = rearrange(K, "batch_size seq_len (num_heads head_dim) -> batch_size num_heads seq_len head_dim", num_heads=self.num_heads, head_dim=self.head_dim)
        V_a = rearrange(V, "batch_size seq_len (num_heads head_dim) -> batch_size num_heads seq_len head_dim", num_heads=self.num_heads, head_dim=self.head_dim)

        if self.rope:
            Q_a = self.rope(Q_a)
            K_a = self.rope(K_a)

        seq_len = x.size(-2)
        mask = torch.tril(torch.ones(seq_len, seq_len, device = x.device, dtype = torch.float32))

        attn_scores = scaled_dot_product_attention(Q_a, K_a, V_a, mask)

        concat_multihead = rearrange(
            attn_scores, "batch_size num_heads seq_len head_dim -> batch_size seq_len (num_heads head_dim)"
        )

        return self.W_o(concat_multihead)