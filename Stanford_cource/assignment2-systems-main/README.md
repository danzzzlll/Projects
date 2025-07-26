# CS336 Spring 2025 – Assignment 2: Systems

**Systems & Parallelism for Language Model Training**

## 📌 Overview

In this assignment, you'll optimize the Transformer model from Assignment 1 for performance on single GPUs and scale it to multi‑GPU training. Your tasks include profiling, kernel optimization, and building efficient distributed training logic.

## 🚀 What I Implemented

scripts for benchmarking is in folder
```
cs336_systems/tasks/
```
Also report on profiling can be found in
```
cs336_systems/tasks/report.md
```

### **Benchmarking & Profiling Tools**
- End-to-end benchmarking of forward/backward passes using Python timing
- Detailed profiling with PyTorch Profiler to analyze compute and memory hot spots
- Memory usage tracking during training and inference  
- Profile GPU usage with nsys.