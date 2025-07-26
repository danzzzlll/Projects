# Online Softmax

## What is Softmax?

The **softmax function** turns a vector of real values into a probability distribution. For a vector $$x \in \mathbb{R}^n,$$ the softmax of element \( x_i \) is:

$$
\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_{j=1}^n e^{x_j}}.
$$

This naive version is **numerically unstable**, especially when elements of \( x \) are large.

To improve numerical stability, we subtract the maximum value from all elements:

$$
\text{softmax}(x_i) = \frac{e^{x_i - \max_j x_j}}{\sum_{j=1}^n e^{x_j - \max_k x_k}}
$$

However, this requires **two passes** over the data:
- One to compute max(x)
- One to compute exponentials and their sum

---

## What is Online Softmax?

**Online Softmax** is a **numerically stable, one-pass** algorithm for computing softmax over a sequence of values — without needing to store the entire input or make multiple passes.

This is especially useful for:
- Large inputs or streaming data
- GPU execution (limited shared memory)
- Fused kernels (e.g. in Triton)

---

## Online Algorithm: Step-by-Step

Let the input be a sequence $$x_1, x_2, \ldots, x_n.$$

We maintain two **running values**:
- \( m_t \): the running maximum after \( t \) elements
- \( d_t \): the running denominator (adjusted sum of exponentials)

### Initialization:

$$
m_0 = -\infty \\
d_0 = 0
$$

### Iteration (for each \( x_t \)):

$$
m_t = \max(m_{t-1}, x_t)
$$

$$
d_t = d_{t-1} \cdot e^{m_{t-1} - m_t} + e^{x_t - m_t}
$$

After processing the full sequence:
- \( m_n \) is the global max
- \( d_n \) is the normalized denominator

The softmax for each element \( x_t \) is then computed as:

$$
\text{softmax}(x_t) = \frac{e^{x_t - m_n}}{d_n}
$$

This can be done either in-place or with a second pass.

---

## Benefits

- **Numerical stability** (like 2-pass version)
- **One-pass** algorithm (good for streaming / GPU)
- **No temporary buffers** needed
- **GPU-efficient**: avoids global memory reads or two sweeps

---

## Comparison

| Feature               | Naive Softmax | Stable Softmax | Online Softmax |
|-----------------------|---------------|----------------|----------------|
| Numerical Stability   | ❌            | ✅             | ✅             |
| One-Pass              | ❌            | ❌             | ✅             |
| Memory-Efficient      | ❌            | ❌             | ✅             |
| GPU Parallelization   | ⚠️ Moderate   | ⚠️ Moderate    | ✅ Efficient   |

---

## Implementations

This section contains several custom implementations of numerically stable softmax using online and block-wise approaches. Each version demonstrates how to compute softmax incrementally or in parallel blocks for improved performance and stability.

---

### `online_softmax_1d(X)`

Implements online softmax for a **single 1D tensor**. Computes softmax for the full vector as if the last element arrived after all others. Useful to demonstrate how online softmax behaves in pure 1D setting.

---

### `block_softmax(X)`

Performs **softmax over a vector split into row-wise blocks** (e.g., 2 blocks of size 4). Each block computes local max and local sum, and then combines them using a global max and rescaled sums. This is useful for simulating how softmax might be computed in **data-parallel** settings across GPU blocks.

---

### `online_block_softmax(X)`

Same as `block_softmax`, but updates max and denominator **incrementally in a streaming fashion**. Maintains global max and accumulates the sum over blocks online. This matches the **classic online softmax** formula but across batches of rows.

---

### `batch_softmax(X)`

Performs **online softmax over feature dimension** (columns) for a batch of vectors. The input tensor `X` of shape `[B, D]` is split into two halves along the last dimension. Softmax is computed using the online reduction logic with max tracking and exponential accumulation. This simulates a 2-block parallel softmax.

---

### `online_batch_softmax(X)`

A **fully online version of batch softmax**, processing blocks of features (columns) incrementally. Maintains running max `M_old` and denominator `L_old` across feature blocks using the numerically stable formula:

$$
M_{\text{new}} = \max(M_{\text{old}}, M) \\
L_{\text{new}} = L_{\text{old}} \cdot e^{M_{\text{old}} - M_{\text{new}}} + \sum e^{x - M_{\text{new}}}
$$

Applies to each batch independently.

---