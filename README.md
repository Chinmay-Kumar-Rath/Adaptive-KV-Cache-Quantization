# Adaptive KV Cache Quantization Simulation

A simulation of importance-based adaptive quantization for transformer KV caches, inspired by research in efficient LLM inference (TurboQuant). Tested on both random weight matrices and real GPT-2 embeddings.

---

## Motivation

In transformer models, to avoid recomputing previous token representations at every step, the model stores them in a **KV Cache**. As the context grows longer, this cache consumes significant memory — a real bottleneck in production LLM inference.

The key insight: **not all tokens are equally important**. Tokens that receive more cumulative attention from subsequent tokens matter more for output quality. Less important tokens can be stored at lower precision without meaningfully affecting results.


---

## How It Works

Each token in the KV cache is assigned a precision level based on its cumulative attention score:

| Rank | Tokens | Precision | Memory Per Element |
|------|--------|-----------|-------------------|
| Top 40% | Most attended | FP32 | 4 bytes |
| Middle 20% | Moderately attended | INT8 | 1 byte |
| Bottom 40% | Least attended | INT4 | 0.5 bytes |

Importance scores accumulate over time using exponential decay:

```
importance[i] = 0.95 * importance[i] + attention_weight[i]
```

This means recent attention matters more than old attention — a token attended to consistently stays important, while one attended to only early on gradually loses priority.

> **Note:** This is a simulation. Tensors are dequantized back to FP32 for computation. Memory savings reported are theoretical estimates of what actual compressed storage would achieve.

---

## Project Structure

| File | Purpose |
|------|---------|
| `quantize.py` | Quantize and dequantize tensors at a given bit precision |
| `KV_cache.py` | KV Cache class with cumulative importance tracking |
| `adaptiveQuantization.py` | Assigns FP32/INT8/INT4 precision based on importance rank |
| `benchmark.py` | Calculates theoretical memory savings and MSE |
| `main.py` | Simulation with random weight matrices — no extra dependencies |
| `main_using_GPT2.py` | Simulation using real GPT-2 token embeddings |
| `graphical_analysis.py` | Generates memory savings and MSE plots |
| `attention.py` | Standalone attention mechanism reference implementation |

---

## How To Run

Install dependencies:
```bash
pip install -r requirements.txt
```

**Quick run — random weights, no extra dependencies:**
```bash
python main.py
```

**Run with real GPT-2 embeddings:**
```bash
python main_using_GPT2.py
```

---

## Results

### Random Weights (dim=4000, 100 tokens)

Memory savings stabilize around **50%** as token count grows. MSE is near zero for most steps but spikes occasionally due to large value ranges in unnormalized random tensors — an expected limitation of naive per-tensor quantization without normalization.

![Random Weights Results](Figure_4Results_With_100Tokens(Random_numbers).png)

### GPT-2 Embeddings (dim=768, 4500 real tokens)

With real GPT-2 embeddings and scaled weight matrices, results are significantly cleaner.

| Metric | Value |
|--------|-------|
| Memory saved (stable) | ~50% |
| Relative MSE (peak) | ~0.0047 |
| Relative MSE (stable) | ~0.0007 |

MSE starts high for the first few tokens (cache is small, importance scores not yet stable) then drops and stabilizes below 0.001 — less than 0.1% relative error despite 50% memory reduction.

![GPT-2 Results](Figure_1Results_With_4500Tokens(In_GPT2).png)

---

## Key Observations

**Memory savings are consistent and predictable.**
Regardless of input type, savings stabilize around 50% — directly determined by the 40/20/40 precision split.

**MSE behaviour depends heavily on input distribution.**
Random unnormalized inputs cause occasional large MSE spikes. Real model embeddings with scaled projections produce stable, near-zero MSE. This highlights why production quantization systems use layer normalization and calibration.

**Decay-based importance is more realistic than simple accumulation.**
Early tokens do not dominate importance forever. Recent attention patterns influence compression decisions more than distant history.

---

## Limitations

- **Simulation only.** No actual RAM is saved — tensors remain FP32 in memory. Savings are theoretical calculations.
- **No layer normalization.** Real transformer KV caches use normalized activations. This simulation skips normalization, which exaggerates MSE on random inputs.
- **Per-tensor quantization.** Production systems use per-channel or per-row quantization for better accuracy. This simulation uses a single scale per token vector.
- **Single-head attention.** Real transformers use multi-head attention. This simulation uses a single attention head.

---

## Future Work (Version 2)

- Actually store INT8/INT4 tensors in the cache with scale and min metadata
- Dequantize only at attention computation time
- Measure real RAM savings using `torch.cuda.memory_allocated()`
- Add layer normalization before quantization to reduce MSE spikes
- Extend to multi-head attention

---

## Inspiration

Inspired by [TurboQuant](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/) and related work on KV cache compression for efficient LLM inference.
