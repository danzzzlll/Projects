import sys
import os
import timeit
import numpy as np
import pandas as pd
import torch
import torch.cuda.nvtx as nvtx

sys.path.append("../../")

from cs336_basics.cs336_basics.model import BasicsTransformerLM
from cs336_basics.cs336_basics.optimizer import get_cosine_lr, AdamW
from cs336_basics.cs336_basics.data import get_batch
from cs336_basics.cs336_basics.nn_utils import cross_entropy
from arguments_utils import parse_args, MODEL_SIZES

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)


def generate_data(args):
    dataset = np.random.randint(0, args.vocab_size, size=args.context_length * 10)
    x, y = get_batch(dataset, batch_size=args.batch_size, context_length=args.context_length, device=args.device)
    return x, y


def bench_loop(transformer_params, optimizer_params, train_params, args):
    model = BasicsTransformerLM(**transformer_params).to(train_params["device"])
    optimizer = AdamW(model.parameters(), **optimizer_params)
    profile = []

    x, y = generate_data(args)

    has_started_memory_recording = False

    for it in range(train_params["num_iters"]):
        tag = "warmup" if it < train_params["warmup_iters"] else "train_iter"

        if not has_started_memory_recording and it == train_params["warmup_iters"]:
            print(">>> Starting CUDA memory history recording")
            torch.cuda.memory._record_memory_history(max_entries=1_000_000)
            has_started_memory_recording = True

        with nvtx.range(tag):
            bench_step = bench_one_step(
                it, model, optimizer, x, y,
                model_size=train_params["model_size"],
                device=train_params["device"],
                is_warmup=(it < train_params["warmup_iters"]),
                only_forward=train_params["only_forward"],
                mixed_precision=train_params["mixed_precision"]
            )
        print(bench_step)
        profile.append(bench_step)

    if has_started_memory_recording:
        print(">>> Dumping CUDA memory snapshot")
        torch.cuda.memory._dump_snapshot("../benches/small-full-512-mixed.pickle")
        torch.cuda.memory._record_memory_history(enabled=None)

    return profile


def bench_one_step(it, model, optimizer, x, y, model_size, only_forward=False, device=None, is_warmup=False, mixed_precision=False):
    dtype = torch.float16 if mixed_precision else torch.float32
    with torch.autocast(device_type=device, dtype=dtype):
        forward_start = timeit.default_timer()
        with nvtx.range("forward pass"):
            logits = model(x)
        torch.cuda.synchronize()
        forward_end = timeit.default_timer()

        if only_forward:
            return {
                "iteration": it,
                "time_forward": forward_end - forward_start,
                "time_backward": 0,
                "time_full": 0,
                "loss": 0,
                "device": device,
                "is_warmup": is_warmup,
                "model_size": model_size,
                "memory_allocated": torch.cuda.memory_allocated(device) if device=="cuda" else 0,
                # "max_memory_allocated": torch.cuda.max_memory_allocated(device),
            }

        backward_start = timeit.default_timer()
        loss = cross_entropy(logits, y)

    with nvtx.range("backward pass"):
        loss.backward()
    torch.cuda.synchronize()
    backward_end = timeit.default_timer()

    with nvtx.range("optimizer step"):
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    full_step_end = timeit.default_timer()

    return {
        "iteration": it,
        "time_forward": forward_end - forward_start,
        "time_backward": backward_end - backward_start,
        "time_full": full_step_end - forward_start,
        "loss": loss.detach().item(),
        "device": device,
        "is_warmup": is_warmup,
        "model_size": model_size,
        "memory_allocated": torch.cuda.memory_allocated(device) if device=="cuda" else 0,
        # "max_memory_allocated": torch.cuda.max_memory_allocated(device),
    }


if __name__ == "__main__":
    args = parse_args()
    print(args)

    transformer_params = MODEL_SIZES[args.model_size]
    transformer_params["vocab_size"] = args.vocab_size
    transformer_params["context_length"] = args.context_length
    transformer_params["rope_theta"] = args.rope_theta
    print(transformer_params)

    optimizer_params = {
        "lr": 1e-3,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "weight_decay": 0.0
    }

    train_params = {
        "num_iters": args.num_iters,
        "warmup_iters": args.warmup_iters,
        "only_forward": args.only_forward,
        "batch_size": args.batch_size,
        "device": args.device,
        "model_size": args.model_size,
        "mixed_precision": args.mixed_precision
    }

    profile = bench_loop(transformer_params, optimizer_params, train_params, args)

    file_name = args.bench_name
    # os.makedirs("benches", exist_ok=True)
    with open(f"../benches/{file_name}", "w") as f:
        f.write(pd.DataFrame(profile).to_markdown())
        print(f"SAVE metrics to benches/{file_name}")
