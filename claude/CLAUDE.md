# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project investigating the **Adaptive Relational Bottleneck** - a hybrid attention mechanism that interpolates between self-attention and relational bottleneck using a learnable parameter γ (gamma):

$$\text{softmax}\left( \frac{(XW_q)(XW_k)^\top}{\sqrt{d_k}} \right)\left[(1-\gamma)X+\gamma I\right]W_v$$

- γ=0 → pure self-attention
- γ=1 → pure relational bottleneck

## Development Commands

```bash
# Install dependencies (uses uv package manager)
uv sync

# Run training scripts
uv run python scripts/block_adaptive_bottleneck.py
uv run python scripts/context_identification_bottleneck.py
uv run python scripts/meta_opt_gamma.py
uv run python scripts/online_opt_gamma.py

# Run Jupyter notebooks
uv run jupyter notebook
```

## Architecture

### Source Modules (`src/adaptive_bottleneck/`)

- **models.py** - Core attention mechanisms with gamma interpolation
  - `init_gamma_value_attn_params()` - parameter initialization
  - `gamma_value_self_attention_forward()` - self-attention variant
  - `gamma_value_cross_attention_forward()` - cross-attention variant (separate queries/context/keys)
  - `forward_logits()` / `forward_multinomial_logits()` - prediction heads

- **tasks.py** - Synthetic evaluation tasks
  - `BlockComparison` - binary classification: do all vectors have 1s in the same block?
  - `ContextIdentification` - multi-class: identify which context item is closest to reference (feature or position mode)

- **analysis.py** - Visualization of learned attention weights (W_q, W_k, W_v matrices, attention scores)

- **utils.py** - Timing decorator, plotting config, directory utilities

- **save.py** - JAX parameter serialization (JSON for results, pickle for pytrees)

### Scripts (`scripts/`)

Training scripts perform gamma sweeps across [0, 1] to analyze where self-attention vs relational bottleneck performs better. Each script follows the pattern:
1. Initialize params with JAX PRNG keys
2. Setup Optax optimizer (AdamW)
3. JIT-compile training step
4. Sweep gamma values, save results to `data/`

### Notebooks (`notebooks/`)

Analysis notebooks visualize hyperparameter sweep results and learned model weights. Results saved to `results/`.

## Key Patterns

- **JAX-first**: All array operations use JAX. Use `jax.random.key()` for reproducibility.
- **Optax for optimization**: AdamW with weight decay is the standard optimizer.
- **Data generation**: Tasks generate synthetic data with constructive sampling to avoid degenerate cases.
- **Serialization**: Use `jax.device_get()` before pickling JAX arrays.
