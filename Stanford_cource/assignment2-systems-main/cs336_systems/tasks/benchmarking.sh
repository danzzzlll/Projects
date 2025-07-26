#!/bin/bash

set -e

# benchmarking_script
python benchmarking_script.py --device=cpu --num_iters=15 --warmup_iters=5 --model_size=tiniest --bench_name=tiniest-cpu-full-fp32.md
python benchmarking_script.py --device=cuda --num_iters=15 --warmup_iters=5 --model_size=tiniest --bench_name=tiniest-cuda-full-fp32.md
## tiny
python benchmarking_script.py --device=cpu --num_iters=15 --warmup_iters=5 --model_size=tiny --bench_name=tiny-cpu-full-fp32.md
python benchmarking_script.py --device=cuda --num_iters=15 --warmup_iters=5 --model_size=tiny --bench_name=tiny-cuda-full-fp32.md
## small
python benchmarking_script.py --device=cpu --num_iters=15 --warmup_iters=5 --model_size=small --bench_name=small-cpu-full-fp32.md
python benchmarking_script.py --device=cuda --num_iters=15 --warmup_iters=5 --model_size=small --bench_name=small-cuda-full-fp32.md


# nsys profile -o benches/small_256_forward python .\benchmarking_script.py --device=cuda --num_iters=15 --warmup_iters=5 --context_length=256 

# memory_profiling
python benchmarking_script.py --device=cuda --num_iters=15 --warmup_iters=5 --model_size=small --bench_name=small-cuda-full-fp32.md --context_length=512 --mixed_precision
python benchmarking_script.py --device=cuda --num_iters=15 --warmup_iters=5 --model_size=small --bench_name=small-cuda-full-fp32.md --context_length=512 --mixed_precision --only_forward

python benchmarking_script.py --device=cuda --num_iters=15 --warmup_iters=5 --model_size=small --bench_name=small-cuda-full-fp32.md --context_length=256 --mixed_precision
python benchmarking_script.py --device=cuda --num_iters=15 --warmup_iters=5 --model_size=small --bench_name=small-cuda-full-fp32.md --context_length=256 --mixed_precision --only_forward

python benchmarking_script.py --device=cuda --num_iters=15 --warmup_iters=5 --model_size=small --bench_name=small-cuda-full-fp32.md --context_length=128 --mixed_precision
python benchmarking_script.py --device=cuda --num_iters=15 --warmup_iters=5 --model_size=small --bench_name=small-cuda-full-fp32.md --context_length=128 --mixed_precision --only_forward

