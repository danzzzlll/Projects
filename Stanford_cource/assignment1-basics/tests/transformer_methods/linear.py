import math
import torch
import torch.nn as nn
from einops import einsum

class Linear(nn.Module):
    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        # in_features: int final dimension of the input
        # out_features: int final dimension of the output
        # device: torch.device | None = None Device to store the parameters on
        # dtype: torch.dtype | None = None Data type of the parameters
        self.device = device

        sigma = math.sqrt(2 / (in_features + out_features))
        min_, max_ = -3 * sigma, 3 * sigma

        weights = torch.empty((in_features, out_features), device=device, dtype=dtype)
        nn.init.trunc_normal_(weights, mean=0.0, std=sigma, a=min_, b=max_)
        self.W = nn.Parameter(weights)

    def forward(self, x:torch.Tensor):
        if self.device is not None:
            x = x.to(self.device)
        output = einsum(
            x, self.W.to(x.dtype), "... in, in out -> ... out"
        )
        return output
    

########################################---TESTS---###############################################
def test_Linear():
    # 1. Проверка форматов
    lin = Linear(2, 5, dtype=torch.float32)
    x = torch.rand((10, 2))  # 2D input
    assert lin(x).shape == (10, 5), "Ошибка для 2D входа"
    
    x = torch.rand((3, 10, 2))  # 3D input
    assert lin(x).shape == (3, 10, 5), "Ошибка для 3D входа"
    
    # 2. Проверка устройства
    if torch.cuda.is_available():
        lin_cuda = Linear(2, 5, device="cuda")
        x_cpu = torch.rand((10, 2))
        assert lin_cuda(x_cpu).device.type == "cuda", "Ошибка переноса на CUDA"
    
    # 3. Проверка dtype
    lin_fp16 = Linear(2, 5, dtype=torch.float16)
    x_fp32 = torch.rand((10, 2), dtype=torch.float32)
    assert lin_fp16(x_fp32).dtype == torch.float32, "Dtype не сохранился (должен остаться float32, так как входной тензор float32)"

    lin_fp32 = Linear(2, 5, dtype=torch.float32)
    x_fp16 = torch.rand((10, 2), dtype=torch.float16)
    assert lin_fp32(x_fp16).dtype == torch.float16, "Ожидается float16!"
    
    print("Все тесты пройдены!")

test_Linear()