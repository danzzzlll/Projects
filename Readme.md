# Repository Guide

## projects

Folder | What’s inside | 
-------|--------------|
`CustomReranker/` | Custom post‑ranking module that plugs a **CatBoost** model into **Llama‑Index**’s `BaseNodePostprocessor`. Contains a serialised `.cbm` model, Python wrapper class, and a demo notebook that shows how to call the reranker on retrieved nodes. 
`NLP/` | Covering NLP models: Transformer, word2vec, rnn, also speculative decoding.
`OnlineSoftmax/` | A collection of numerically‑stable softmax implementations (PyTorch + Triton): 1‑D streaming softmax, block‑wise batched softmax, and GPU kernels for online reduction.
`Stanford_cource/` | Assignments - 1. for full training pipeline of GPT2 with modern improvements such as pre-layer norms, swiglu, byte-bpe.
`triton_kernels/` | Stand‑alone Triton GPU kernels (ReLU, online softmax, weighted sum).
`markov_chain/`  | Implemenation of simple Markov chain fro generating text based on Master And Margarite book.
`deep-ml.ipynb` | A single notebook of lecture notes / experiments from various deep‑ML readings (optimisers, weight decay schedules, etc.). 
---

