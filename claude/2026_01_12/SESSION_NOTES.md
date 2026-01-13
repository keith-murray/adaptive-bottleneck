# Session Notes: 2026-01-12

## Goal
Implement learnable γ (via backpropagation) for ContextIdentification task across all three label modes: feature, position, and both.

## What We Did

### 1. Created Learnable Gamma Sweep Script
**Branch**: `sweep`

- Implemented unified script `scripts/context_identification/learnable_gamma_sweep.py`
- Handles all 3 label modes:
  - **Feature**: single decoder predicting which feature is closest
  - **Position**: single decoder predicting which position is closest
  - **Both**: dual decoders predicting both feature and position
- Sweep configuration:
  - γ initialization: {0.0, 0.25, 0.5, 0.75, 1.0}
  - Seeds: 5 per initialization
  - Total: 75 runs (5 init × 5 seeds × 3 modes)
- Softmax attention only (no linear)
- Separate SGD optimizer for γ with clipping to [0,1]
- Tracks γ evolution throughout training

### 2. Fixed Critical Performance Issue
**Commit**: `d6ee54c`

**Problem**: Initial sweep took ~4 hours due to JIT recompilation
- Eval step was being recreated every epoch with updated γ value
- JAX recompiled 2500 times per run × 75 runs = massive slowdown

**Solution**:
- Changed `make_eval_step` functions to accept γ as function argument (not closure)
- Create eval_step once before training loop
- Pass `gamma_param` to eval step each call
- **Result**: Sweep time reduced from 4 hours → 20 minutes

### 3. Optimizer Discovery
**Commit**: `905ed41`

**Finding**: Position task performed poorly with AdamW optimizer for γ

**Solution**: Switched γ optimizer from AdamW to SGD
```python
# Before
gamma_optimizer = optax.adamw(learning_rate=gamma_lr, weight_decay=0.0)

# After
gamma_optimizer = optax.sgd(learning_rate=gamma_lr)
```

**Why SGD works better for γ**:
- Single scalar parameter - AdamW's adaptive mechanisms become a liability
- AdamW's momentum can overshoot optimal γ values
- Position task likely has noisier gradients w.r.t. γ
- SGD's simplicity is ideal for low-dimensional optimization
- Result: γ successfully converges to expected values

### 4. Visualization and Analysis
**Commit**: `e415511`

Created `notebooks/plot_learned_gamma_sweep.ipynb` with:
- 3-panel plots (one per label mode)
- Test accuracy evolution showing mean ± SEM across seeds
- γ trajectory plots showing parameter evolution
- Visual improvements:
  - Colormap: viridis → plasma
  - Selective legends (only on feature and both panels)
  - Compact layout (15×4 → 12.5×3)

**Results**:
- Feature mode: γ → 0 (pure self-attention)
- Position mode: γ → 1 (pure relational bottleneck)
- Both mode: γ → intermediate values (balanced representation)

### 5. Pull Request
**PR #4**: https://github.com/keith-murray/adaptive-bottleneck/pull/4
- Created and merged into main
- Branch `sweep` deleted after merge

### 6. Documentation: Signed Interpolation Walkthrough
**File**: `docs/signed_interpolation_walkthrough/signed_interpolation_walkthrough.tex`

Created LaTeX document explaining why signed interpolation $(1-\gamma)X - \gamma I$ enables dual decoders to solve both feature and position tasks:

**The Problem**:
- Standard interpolation $(1-\gamma)X + \gamma I$ entangles information
- At intermediate γ, feature and position info are mixed in same vector
- Linear decoders cannot separate entangled representations

**The Solution**:
- Signed variant $(1-\gamma)X - \gamma I$ creates separable components
- Positive values encode feature information (from $X$)
- Negative values encode position information (from $-I$)
- Dual decoders learn to extract each component independently

**Key Insight**: Differential encoding via opposite polarities creates linearly separable subspaces, analogous to:
- Signal processing: I/Q modulation
- Neuroscience: ON/OFF cell encoding
- Linear algebra: Orthogonal subspace decomposition

### 7. Partial Relational Results Plotting
**File**: `scripts/partial_relational/plot_partial_relational_results.py`

Created plotting script for `data/partial_relational_sweep.json`:
- 2-panel plot showing feature and position accuracies separately
- Alternative single-panel plot showing averaged accuracy
- Supports both linear and softmax attention modes
- Structure: `data[attn_kind][gamma][seed]`

## Key Learnings

1. **JIT compilation matters**: Always pass dynamic values as function arguments, not closure captures
2. **Optimizer choice for meta-parameters**: For single scalar parameters like γ, simple SGD often outperforms adaptive methods
3. **Signed interpolation is not interpolation**: It's differential encoding that creates separable components through opposite polarities
4. **Visualization workflow**: Jupyter notebooks enable faster iteration than standalone scripts

## Files Created/Modified

**Scripts**:
- `scripts/context_identification/learnable_gamma_sweep.py`
- `scripts/partial_relational/plot_partial_relational_results.py`

**Notebooks**:
- `notebooks/plot_learned_gamma_sweep.ipynb`

**Documentation**:
- `docs/signed_interpolation_walkthrough/signed_interpolation_walkthrough.tex`

**Results**:
- `results/learnable_gamma_accuracy_curves.png`
- `results/learnable_gamma_evolution.png`
- `results/partial_relational_accuracy_curves_softmax.png`
- `results/partial_relational_averaged_accuracy_softmax.png`

**Data**:
- `data/learnable_gamma_sweep.json` (75 runs, 2500 epochs each)
- `data/learnable_gamma_models/` (saved model parameters)

## Next Steps

Potential directions:
1. Analyze final γ values across different initializations and label modes
2. Compare learnable γ results with fixed γ sweeps
3. Investigate whether intermediate γ values in "both" mode correspond to signed interpolation predictions
4. Extend to other tasks or architectures
