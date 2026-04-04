# Session Notes: 2026-04-04

## Goal
Investigate whether the online SGD regime (γ₀=0.5, SGD for γ, AdamW for weights) that worked for ContextIdentification also allows ARCA to learn the XOR task.

## Background
Previous work (2026-03-20) showed that both γ=0 and γ=1 can solve XOR in a fixed-γ setting, unlike ContextIdentification where each endpoint uniquely solves one label mode. The question now is whether ARCA with learnable γ can reliably discover the correct solution.

## What We Did

### 1. Repo Reorganization
Reorganized `notebooks/`, `results/`, and `scripts/context_identification/` into task- and experiment-based subdirectories for easier navigation:

- `notebooks/` → `block/`, `context_identification/`, `figures/`
- `results/context_identification/` → `fixed_gamma/`, `learnable_gamma/` (→ renamed `online_gamma/`), `meta_gamma/`, `lagrange_gamma/`, `online_gamma/`, `pos_codes/`
- `scripts/context_identification/` → `fixed_gamma/`, `learnable_gamma/`, `meta_gamma/`, `lagrange_gamma/`, `online_gamma/`, `pos_codes/`

Key distinction: `meta_gamma`, `online_gamma`, and `lagrange_gamma` are three separate γ optimization strategies, not variants of the same thing.

### 2. Online Gamma Comparison: Indexing vs Decoder
**File**: `scripts/xor/online_gamma_comparison.py`
**Branch**: `feature/xor-online-gamma-comparison`
**Commit**: `1b4d4b2`

Compared two architectural readout variants for XOR under three γ conditions (RCA, cross-attention, ARCA):
- **Indexing**: `out[:, 0, :]` → `(B, 2)` logits, CE loss (matches ContextIdentification architecture)
- **Decoder**: flattened attention output → linear classifier → scalar logit, BCE loss

**Architectural detour — cross-attention**:
We initially tried cross-attention (`gamma_value_cross_attention_forward`) splitting XOR inputs as query/context. This failed for all conditions. The reason: with a length-1 context, softmax reduces to a constant weight of 1.0, so the query never influences the output. At γ=1, the identity matrix for a length-1 context degenerates to `[[1, 0]]`, making the values a constant independent of both inputs. Self-attention is required because both tokens must interact through the attention weight matrix.

**Results** (final epoch, 20 seeds, 500 epochs):
| Arch      | Condition   | Acc           | γ_final       |
|-----------|-------------|---------------|---------------|
| indexing  | rca         | 1.000 ± 0.000 | 1.000 ± 0.000 |
| indexing  | cross_attn  | 0.755 ± 0.017 | 0.000 ± 0.000 |
| indexing  | arca        | 0.914 ± 0.118 | 0.653 ± 0.473 |
| decoder   | rca         | 1.000 ± 0.000 | 1.000 ± 0.000 |
| decoder   | cross_attn  | 0.754 ± 0.018 | 0.000 ± 0.000 |
| decoder   | arca        | 0.829 ± 0.113 | 0.311 ± 0.451 |

The large std on ARCA γ_final (±0.47) indicated bimodality — seeds splitting between the γ=0 and γ=1 basins.

### 3. Gamma Initialization Comparison
**File**: `scripts/xor/init_gamma_comparison.py`
**Commit**: `9e7816e`

Replaced the three γ conditions with ARCA at three initializations {0.0, 0.5, 1.0} to test whether initialization determines which basin ARCA falls into.

**Results** (final epoch, 20 seeds, 2500 epochs):
| Arch      | γ₀  | Acc           | γ_final       |
|-----------|-----|---------------|---------------|
| indexing  | 0.0 | 0.756 ± 0.017 | 0.014 ± 0.016 |
| indexing  | 0.5 | 0.901 ± 0.122 | 0.606 ± 0.482 |
| indexing  | 1.0 | 1.000 ± 0.000 | 1.000 ± 0.000 |
| decoder   | 0.0 | 0.752 ± 0.015 | 0.013 ± 0.013 |
| decoder   | 0.5 | 0.792 ± 0.088 | 0.161 ± 0.353 |
| decoder   | 1.0 | 1.000 ± 0.000 | 1.000 ± 0.000 |

## Key Learnings

1. **Self-attention is required for XOR**: Cross-attention with a length-1 context degenerates — the attention weight is always 1.0, so the query has no influence on the output. XOR requires both inputs to interact through the attention weight matrix, which only self-attention provides.

2. **The XOR loss landscape has two basins**: γ=0 and γ=1 are both local attractors. Once weights adapt to one regime, γ gets no gradient to escape.

3. **Initialization determines the outcome**: γ₀=0.0 always gets stuck; γ₀=1.0 always succeeds; γ₀=0.5 is bimodal (seeds split between basins), with the `indexing` architecture showing a stronger pull toward γ=1 than `decoder`.

4. **Indexing architecture gives cleaner gradients for γ**: The CE loss in the indexing setup provides stronger gradient signal for γ than BCE in the decoder setup, as seen by the better performance at γ₀=0.5.

## Next Steps

1. Understand *why* γ=0 is a stable attractor for XOR — what solution does the cross-attention model find that achieves ~75% accuracy?
2. Investigate whether a different γ optimizer or learning rate schedule can help ARCA escape the γ=0 basin from γ₀=0.5.
3. Consider whether the bimodality result is a feature (the task has two valid computational strategies) or a bug (optimization is unreliable).
