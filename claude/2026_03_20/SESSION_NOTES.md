# Session Notes: 2026-03-20

## Goal
Implement the XOR task and train the adaptive bottleneck on it for both fixed and learnable γ values.

## Background
Previous work validated that the adaptive bottleneck can accommodate different *label types* (feature identity vs. position) within the `ContextIdentification` task. The XOR task is meant to push beyond labeling — it requires computing a relational function over two inputs, neither of which individually determines the output.

## What We Did

### 1. Implemented the XOR Task
**File**: `src/adaptive_bottleneck/tasks.py`

Added `XOR` class following the same interface as `BlockComparison` and `ContextIdentification`:
- **Input**: `(B, 2, 2)` — two 2-dimensional one-hot vectors
  - "false" = `[1, 0]`, "true" = `[0, 1]`
  - e.g., `[[1,0], [0,1]]` = (False, True) → XOR = True
- **Label**: `(B,)` float32 — 1.0 if inputs differ, 0.0 if same
- Randomly samples x1, x2 ∈ {0, 1}, computes XOR label, one-hot encodes
- Same `generate_jax_tensors` / `generate_dataset` interface using `jdl.DataLoader`

### 2. Fixed γ Sweep Script
**File**: `scripts/xor/xor_bottleneck.py`

Follows the `context_identification_bottleneck.py` pattern:
- Uses `forward_logits` from `models.py` (self-attention + binary linear classifier)
- BCE loss + sigmoid accuracy (binary classification, consistent with `BlockComparison`)
- Model: `in_dim=2, attn_dim=2, val_dim=2` + classifier `val_dim=2, seq_len=2`
- Sweep: `seeds × gammas × attn_kinds`, saves to `data/xor_gamma_sweep.json`
- Main config: 5 seeds × 5 γ values × 2 attn kinds, 2500 epochs

### 3. Learnable γ Sweep Script
**File**: `scripts/xor/learnable_gamma_sweep.py`

Follows the `context_identification/learnable_gamma_sweep.py` pattern:
- AdamW for model params, SGD for γ (same optimizer split as before)
- γ clipped to [0, 1] after each update
- Tracks γ trajectory in history under `"gamma"` key
- Sweep: `seeds × init_gammas × attn_kinds`, saves to `data/xor_learnable_gamma_sweep.json`

### 4. Plotting Script
**File**: `scripts/xor/plot_xor_results.py`

Three figures, all using `load_plot_config()`, plasma colormap, SEM shading:

1. **`results/xor_gamma_sweep.png`** — fixed-γ test accuracy curves
   - 1×2 panels (linear, softmax), lines colored by γ, shared colorbar

2. **`results/xor_learnable_gamma_sweep.png`** — learnable-γ dynamics
   - 2×2 grid: top row = test accuracy, bottom row = γ trajectory; cols = linear/softmax

3. **`results/xor_fixed_vs_learnable.png`** — direct comparison (softmax only)
   - 1×2 panels: fixed γ vs learnable γ (same colormap, same γ values)
   - Shows whether learning γ improves performance over fixing it

### 5. Commits
- `8d109ec` — XOR task class and training scripts
- `3ac08d3` — plotting script and figures

## Key Learnings

1. **XOR is solvable by both γ=0 and γ=1**: Unlike `ContextIdentification`, XOR doesn't cleanly favor one endpoint. Self-attention (γ=0) can use the attended value mix to encode "same vs different"; the relational bottleneck (γ=1) can use attention weights over position markers to do the same.
2. **XOR goes beyond labeling**: The correct output depends on the *relationship* between two inputs, not on either input individually — making it a test of the model's ability to compute relational functions rather than just retrieve stored labels.

## Next Steps

Potential directions:
1. Analyze the learned γ values — do they converge consistently, or is XOR indifferent to γ?
2. Compare the learned attention weights at γ=0 vs γ=1 to understand the two solution strategies
3. Consider tasks where the two inputs have more structure (e.g., multi-bit XOR, parity)
