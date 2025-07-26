from jaxtyping import Float
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import timeit
import numpy as np
import torch
import torch.cuda.nvtx as nvtx

import pandas as pd

import sys
sys.path.append("../../")

from cs336_basics.cs336_basics.nn_utils import *
from cs336_basics.cs336_basics.optimizer import AdamW
from cs336_basics.cs336_basics.model import *

import argparse

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

batch_size = 8


class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.output_proj = Linear(self.d_model, self.d_model)

    def forward(self, Q, K, V) -> Float[Tensor, " ... seq d_v"]:
        attn_output = scaled_dot_product_attention(K=K, Q=Q, V=V)
        output = self.output_proj(attn_output)
        return output


def pytorch_attention_bench(args, model_dims, seq_lens):
    profile = []

    for model_dim in model_dims:
        for seq_len in seq_lens:
            print(f"Benchmarking d_model: {model_dim}, seq_len: {seq_len}")

            attn = CausalMultiHeadSelfAttention(model_dim).to(args.device)
            optimizer = AdamW(attn.parameters())

            q = torch.rand((args.batch_size, seq_len, model_dim), device=args.device)
            k = torch.rand((args.batch_size, seq_len, model_dim), device=args.device)
            v = torch.rand((args.batch_size, seq_len, model_dim), device=args.device)
            y = torch.randn((args.batch_size, seq_len, model_dim), device=args.device)

            for _ in range(args.warmup_iters):
                with torch.no_grad(): 
                    _ = attn(q, k, v)
                if not args.only_forward:
                    
                    loss_warmup = F.mse_loss(attn(q, k, v), y)
                    loss_warmup.backward()
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize()

            torch.cuda.memory._record_memory_history(max_entries=2000)
            try:
                for it in range(args.num_iters - args.warmup_iters): 
                    with nvtx.range(f"Profiled_Iter_{it}_d{model_dim}_s{seq_len}"):
                        torch.cuda.synchronize()
                        forward_start = timeit.default_timer()

                        logits = attn(q, k, v)
                        torch.cuda.synchronize()
                        forward_end = timeit.default_timer()

                        memory_after_forward = torch.cuda.memory_allocated(args.device) / 1024 / 1024

                        time_backward = 0
                        time_full = 0
                        loss_val = 0

                    if not args.only_forward:
                      
                        loss = F.mse_loss(logits, y)
                        loss_val = loss.item()

                        torch.cuda.synchronize()
                        backward_start = timeit.default_timer()
                    
                        loss.backward()
                        torch.cuda.synchronize()
                        memory_after_backward = torch.cuda.memory_allocated(args.device) / 1024 / 1024
                        backward_end = timeit.default_timer()
                        time_backward = backward_end - backward_start

                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True) 
                        torch.cuda.synchronize()
                        full_step_end = timeit.default_timer()
                        time_full = full_step_end - forward_start

                    profile.append({
                        "model_dim": model_dim,
                        "seq_len": seq_len,
                        "iteration_in_sample": it, 
                        "time_forward": forward_end - forward_start,
                        "time_backward": time_backward,
                        "time_full_step": time_full, 
                        "loss": loss_val,
                        "device": args.device,
                        "memory_after_forward": memory_after_forward,
                        "memory_after_backward": memory_after_backward if not args.only_forward else 0.
                    })

            except torch.cuda.OutOfMemoryError:
                print(f"CUDA Out of Memory DURING PROFILING for d_model={model_dim}, seq_len={seq_len}.")
                print("Partial memory profile might be saved.")
                # Очищаем кэш после ошибки
                torch.cuda.empty_cache()
                # Профиль будет автоматически сохранен в finally блоке

            except Exception as e:
                print(f"An error occurred DURING PROFILING for d_model={model_dim}, seq_len={seq_len}: {e}")
                torch.cuda.empty_cache()

            finally:
                # === ОСТАНАВЛИВАЕМ ЗАПИСЬ И СОХРАНЯЕМ СНИМОК ===
                print(">>> Dumping CUDA memory snapshot")
                snapshot_filename = f"memory_profile_d{model_dim}_s{seq_len}_iters{args.num_iters}.pickle"
                torch.cuda.memory._dump_snapshot(snapshot_filename)
                torch.cuda.memory._record_memory_history(enabled=None) # Останавливаем запись
                print(f"Snapshot saved to {snapshot_filename}")

                del attn, optimizer, q, k, v, y, logits
                if 'loss' in locals(): del loss
                if 'loss_warmup' in locals(): del loss_warmup
                torch.cuda.empty_cache()
                print("\nDetailed memory profiling complete.")

    print("--- Benchmark Complete ---")

    df = pd.DataFrame(profile)

    aggregated_results = df.groupby(['model_dim', 'seq_len']).agg({
        'time_forward': 'mean',
        'time_backward': 'mean',
        'time_full_step': 'mean',
        'memory_after_forward': 'mean',
        'memory_after_backward': 'mean',
        'loss': 'last'
    }).reset_index()

    print("\nAggregated Results:")
    print(aggregated_results.to_markdown())

    # Сохраняем агрегированные результаты
    with open(f"attn_bench_aggregated.md", "w") as f:
        f.write(aggregated_results.to_markdown())

    # with open(f"attn_bench.md", "w") as f:
    #     f.write(df.to_markdown())

    return profile




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--seq_len", default=64, type=int)
    parser.add_argument("--d_model", default=16, type=int)
    parser.add_argument("--num_iters", default=100, type=int)
    parser.add_argument("--warmup_iters", default=15, type=int)
    parser.add_argument("--only_forward", action="store_true", help="Run only the forward pass")
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--profile_d_model", type=int, help="Model dimension for detailed memory profiling")
    parser.add_argument("--profile_seq_len", type=int, help="Sequence length for detailed memory profiling")
    parser.add_argument("--profile_iters", type=int, default=1, help="Number of benchmark iterations to include in the memory profile snapshot")

    args, _ = parser.parse_known_args()
    print(args)

    model_dims = [16, 32, 64, 128]
    # model_dims = [16]
    seq_lens = [    256, 512, 1024, 2048, 4096] # 8192, 16384
    # seq_lens = []
    pytorch_attention_bench(args, model_dims, seq_lens)
