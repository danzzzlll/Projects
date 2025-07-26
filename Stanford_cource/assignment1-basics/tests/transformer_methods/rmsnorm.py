import torch
import torch.nn as nn
from einops import einsum, rearrange

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # d_model: int Hidden dimension of the model
        # eps: float = 1e-5 Epsilon value for numerical stability
        # device: torch.device | None = None Device to store the parameters on
        # dtype: torch.dtype | None = None Data type of the parameters
        self.d_model = d_model
        self.dtype = dtype
        self.device = device
        self.eps = eps
        self.gamma = nn.Parameter(torch.rand(d_model, device=device, dtype=dtype), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #(batch_size, sequence_length, d_model)
        if self.device is not None:
            x = x.to(self.device)
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms = rearrange(torch.sqrt(
            einsum(x, x, "... d, ... d -> ...") / self.d_model + self.eps), 
                "... -> ... 1"
        )

        normalized = x / rms

        result = einsum(
            normalized, self.gamma, 
            "... d, d -> ... d"
        )

        return result.to(in_dtype)