# Session Notes: 2026-01-10

## Goal
Validate a "half-open" bottleneck (γ≈0.5) for partially-relational tasks that require both feature and position prediction.

## What We Did

### 1. Setup
- Created `feature/partial-relational-task` branch
- Added `TODO.md` documenting the plan

### 2. Extended ContextIdentification Task
- Added `label_mode="both"` that returns both feature and position labels
- Updated `generate_jax_tensors()` and `generate_dataset()` to handle dual outputs
- Created `scripts/test_both_label_mode.py` sanity check

### 3. Extended Model with Dual Decoders
- Added `init_dual_decoder_params(key, val_dim, num_classes, use_bias=True)`
- Added `dual_decoder_cross_attention_forward()` returning `(logits_f, logits_p, attn_w)`
- Loss function: `CE(ZW_f, f) + CE(ZW_p, p)`

### 4. Created Gamma Sweep Script
- `scripts/partial_relational_bottleneck.py`
- Sweeps γ ∈ {0.0, 0.25, 0.5, 0.75, 1.0} with dual decoders

### 5. Initial Experiment Failed
- Ran sweep with standard interpolation `(1-γ)X + γI`
- **Result**: No γ value could solve both tasks
- **Reason**: Feature and position information become entangled at intermediate γ values; linear decoders cannot separate them

### 6. Signed Interpolation Variant
- Changed to `(1-γ)X - γI` (negative sign on identity term)
- Opposite signs "tag" feature (+) and position (-) information
- Linear decoders can now separate the two

### 7. Signed Sweep Succeeded
Results (softmax attention):
| γ    | Feature | Position |
|------|---------|----------|
| 0.00 | 96%     | 10%      |
| 0.25 | 98%     | 84%      |
| 0.50 | 87%     | 88%      |
| 0.75 | 58%     | 73%      |
| 1.00 | 13%     | 97%      |

γ=0.25-0.5 achieves good accuracy on both tasks.

### 8. Merged PR
- PR #3: https://github.com/keith-murray/adaptive-bottleneck/pull/3

## Key Insight
The signed variant works but is **not a true interpolation** - it's a "signed superposition" or "differential encoding" that creates separable components via opposite polarity, rather than smoothly transitioning between self-attention and relational bottleneck.

## Open Question
What architectures can achieve partially-relational behavior while still using genuine interpolation?
