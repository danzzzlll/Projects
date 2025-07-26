import argparse
# import torch

MODEL_SIZES = {
    "tiniest": {"d_model": 4, "num_layers": 1, "num_heads": 1, "d_ff": 16},
    "tiny": {"d_model": 64, "num_layers": 12, "num_heads": 4, "d_ff": 256},
    "small": {"d_model": 768, "num_layers": 12, "num_heads": 12, "d_ff": 3072},
    "medium": {"d_model": 1024, "num_layers": 24, "num_heads": 16, "d_ff": 4096},
    "large": {"d_model": 1280, "num_layers": 36, "num_heads": 20, "d_ff": 5120},
    "xl": {"d_model": 1600, "num_layers": 48, "num_heads": 25, "d_ff": 6400},
    "2.7B": {"d_model": 2560, "num_layers": 32, "num_heads": 32, "d_ff": 10240},
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=4, type=int)
    parser.add_argument("--vocab_size", default=10000, type=int)
    parser.add_argument("--context_length", default=256, type=int)
    parser.add_argument("--d_model", default=768, type=int)
    parser.add_argument("--num_layers", default=12, type=int)
    parser.add_argument("--num_heads", default=12, type=int)
    parser.add_argument("--d_ff", default=3072, type=int)
    parser.add_argument("--rope_theta", default=10000, type=int)
    parser.add_argument("--num_iters", default=100, type=int)
    parser.add_argument("--warmup_iters", default=50, type=int)
    parser.add_argument("--only_forward", action="store_true", help="Run only the forward pass")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--model_size", type=str, default="small")
    parser.add_argument("--mixed_precision", action="store_true", help="Run in torch.float32")

    parser.add_argument("--bench_name", type=str, default="output.md")
    args, _ = parser.parse_known_args()
    return args


