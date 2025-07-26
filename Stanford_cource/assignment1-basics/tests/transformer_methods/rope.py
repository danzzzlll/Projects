import torch
import torch.nn as nn
from einops import rearrange

class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        # theta: float Θ value for the RoPE
        # d_k: int dimension of query and key vectors
        # max_seq_len: int Maximum sequence length that will be inputted
        # device: torch.device | None = None Device to store the buffer on

        self.device = device
        self.max_seq_len = max_seq_len

        freqs = 1. / (theta ** (torch.arange(0, d_k, 2).float() / d_k)).to(device)
        # self.register_buffer("freqs", freqs, persistent=False)

        max_seq_positions = torch.arange(max_seq_len, device=device).float()
        angle = torch.outer(max_seq_positions, freqs)  # shape: (max_seq_len, dim // 2)

        self.cos = angle.cos()
        self.sin = angle.sin()
        # self.register_buffer("cos", cos)
        # self.register_buffer("sin", sin)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None) -> torch.Tensor:
        seq_len = x.size(-2)
        
        if token_positions is None:
            token_positions = torch.arange(seq_len, device=self.device)

        cos = self.cos[token_positions]
        sin = self.sin[token_positions]

        x1, x2 = x[..., 0::2], x[..., 1::2]

        x_rot_0 = x1 * cos - x2 * sin
        x_rot_1 = x1 * sin + x2 * cos

        x_out = torch.stack([x_rot_0, x_rot_1], dim=-1)
        x_out = rearrange(x_out, '... seq d_half two -> ... seq (d_half two)')

        return x_out