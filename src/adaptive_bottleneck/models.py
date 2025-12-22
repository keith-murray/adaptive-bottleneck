import jax.random as random
import jax.numpy as jnp
import jax.nn as nn


# Attention mechanism
def init_gamma_value_attn_params(key, in_dim, attn_dim, val_dim, scale=0.02):
    """
    Initialize parameters for gamma-value self-attention in pure JAX.

    Shapes:
      W_q:    (in_dim, attn_dim)
      W_k:    (in_dim, attn_dim)
      W_v:    (in_dim, val_dim)
    """
    k1, k2, k3 = random.split(key, 3)

    params = {
        "W_q": scale * random.normal(k1, (in_dim, attn_dim)),
        "W_k": scale * random.normal(k2, (in_dim, attn_dim)),
        "W_v": scale * random.normal(k3, (in_dim, val_dim)),
    }

    return params


def gamma_value_attention_forward(params, X, gamma):
    """
    Pure JAX forward pass.

    Args:
      params: dict from init_gamma_value_attn_params
      X:      (B, L, D_in) input sequence; note that it must be that L==D_in
      gamma:  scalar in [0, 1]

    Returns:
      out:         (B, L, D_val)
      attn_weights:(B, L, L)
    """
    B, L, D_in = X.shape
    D_attn = params["W_q"].shape[1]

    # Projections
    Q = X @ params["W_q"]          # (B, L, D_attn)
    K = X @ params["W_k"]          # (B, L, D_attn)

    # Build identity matrix with shape (B, L, D_in)
    I = jnp.broadcast_to(jnp.eye(L, dtype=X.dtype)[None, :, :], (B, L, D_in))

    # Gamma-interpolated input to values
    X_tilde = (1.0 - gamma) * X + gamma * I  # (B, L, D_in)

    # Values
    V = X_tilde @ params["W_v"]   # (B, L, D_val)

    # Attention weights
    scores = jnp.einsum("bld,bmd->blm", Q, K) / jnp.sqrt(D_attn)   # (B, L, L)
    attn_weights = nn.softmax(scores, axis=-1)

    # Output
    out = jnp.einsum("blm,bmd->bld", attn_weights, V)          # (B, L, D_val)
    return out, attn_weights


# Linear classifier
def init_classifier_params(key, val_dim, seq_len, out_dim=1, scale=0.02):
    k1, k2 = random.split(key)
    W_out = scale * random.normal(k1, (seq_len * val_dim, 1))
    b_out = jnp.zeros((out_dim,))
    return {"W_out": W_out, "b_out": b_out}


def forward_logits(params_attn, params_clf, X, gamma,):
    """
    X: (B, L, D_val)
    returns logits: (B,)
    """
    out, attn_w = gamma_value_attention_forward(
        params_attn, X, gamma=gamma,
    )  # out: (B, L, D_val)

    out_flat = out.reshape(out.shape[0], -1)  # (B, L * D_val)
    bias = jnp.broadcast_to(params_clf["b_out"], (out_flat.shape[0],))  # (B,)
    logits = (out_flat @ params_clf["W_out"]).squeeze(-1) + bias
    return logits, attn_w


def forward_multinomial_logits(params_attn, params_clf, X, gamma,):
    """
    X: (B, L, D_val)
    returns logits: (B,)
    """
    out, attn_w = gamma_value_attention_forward(
        params_attn, X, gamma=gamma,
    )  # out: (B, L, D_val)

    out_flat = out.reshape(out.shape[0], -1)  # (B, L * D_val)
    bias_shape = params_clf["b_out"].shape[-1]
    bias = jnp.broadcast_to(params_clf["b_out"][None,:], (out_flat.shape[0],bias_shape,))  # (B,)
    logits = (out_flat @ params_clf["W_out"]).squeeze(-1) + bias
    return logits, attn_w
