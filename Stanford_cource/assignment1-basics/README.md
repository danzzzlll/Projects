# CS336 Spring 2025 – Assignment 1: Basics

**Language Modeling from Scratch**

# My Realization of GPT2

It can be found in folder
```
tests/transformer_methods/
```
All tests from cource are passed and realization for them is in folder
```
tests/adapters.py
```

## 📌 Overview

In this assignment, I implemented all core components required to train a Transformer-based language model from scratch. The main parts include:

- A byte-level BPE tokenizer
- A Transformer language model
- Cross-entropy loss and AdamW optimizer
- A training loop with checkpointing and text generation

## 🚀 Builded Stages

### 1. **Byte-Pair Encoding (BPE) Tokenizer**
- Initialize vocabulary with all 256 byte values
- Pre-tokenization using regex
- Learn BPE merge rules from the training corpus
- Encode text into token ID sequences
- Report on:
  - Compression ratio (bytes/token)
  - Comparison: TinyStories vs OpenWebText
  - Throughput (bytes/sec)
  - Estimated time to tokenize a large corpus
  - Rationale for using `uint16` for serialization

### 2. **Transformer Language Model**
- Implement full Transformer LM architecture:
  - Parameters: `d_model`, `num_heads`, `d_ff`, `num_layers`
  - Positional and token embeddings
- Use Pre-norm Transformer blocks with:
  - RMSNorm
  - Causal (masked) multi-head self-attention
  - Feedforward layers
- Final linear layer for language modeling (LM head)

### 3. **Cross-Entropy Loss & AdamW Optimizer**
- Implement numerically stable, log-softmax-based cross-entropy
- Mean over batch dimension
- Implement AdamW optimizer from scratch:
  - Correct bias correction
  - Proper step tracking
  - Separate handling for parameters

### 4. **Training Loop**
- Load and preprocess datasets
- Train Transformer model
- Save and load model checkpoints
- Evaluate perplexity on validation data
- Generate text with:
  - Temperature sampling
  - Top-p (nucleus) sampling
  - Max token limits

## 🧪 Running the Code

### Setup

```bash
uv install
uv run <your_script.py>


