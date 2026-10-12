import jax.random as random
import jax.numpy as jnp
import jax.nn as nn


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


# Self attention
def gamma_value_self_attention_forward(params, X, gamma, linear=False):
    """Linear gamma-value attention.

    Args:
      params: dict from init_gamma_value_attn_params
      X:      (B, L, D_in) input sequence; note that it must be that L==D_in
      gamma:  scalar in [0, 1]

    Returns:
      out:         (B, L, D_val)
      attn_weights:(B, L, L)  (raw scores)
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

    # Raw attention weights (no softmax)
    scores = jnp.einsum("bld,bmd->blm", Q, K) / jnp.sqrt(D_attn)   # (B, L, L)
    if linear:
        attn_weights = scores 
    else:
        attn_weights = nn.softmax(scores, axis=-1)

    # Output
    out = jnp.einsum("blm,bmd->bld", attn_weights, V)          # (B, L, D_val)
    return out, attn_weights


# Cross attention
def gamma_value_cross_attention_forward(params, X_context, X_query, gamma, linear=False):
    """
    Cross-attention with gamma-interpolated values:

        Q = X_query   @ W_q
        K = X_context @ W_k
        V = ((1-gamma) * X_context + gamma * I) @ W_v

    Args:
        params: dict with keys {"W_q", "W_k", "W_v"}
            W_q: (D_in, D_attn)
            W_k: (D_in, D_attn)
            W_v: (D_in, D_val)
        X_context: (B, L_c, D_in)
        X_query:   (B, L_q, D_in)
        gamma: scalar in [0, 1]
        linear (bool): If True, skip softmax and use raw scores as attention weights.

    Returns:
        out: (B, L_q, D_val)
        attn_weights: (B, L_q, L_c)
    """
    B, L_c, D_in = X_context.shape
    D_attn = params["W_q"].shape[1]

    # Projections
    Q = X_query   @ params["W_q"]  # (B, L_q, D_attn)
    K = X_context @ params["W_k"]  # (B, L_c, D_attn)

    # Build identity matrix with shape (B, L_c, D_in)
    I_ctx = jnp.broadcast_to(jnp.eye(L_c, D_in, dtype=X_context.dtype)[None, :, :], (B, L_c, D_in))

    # Gamma-interpolated values input
    X_tilde = (1.0 - gamma) * X_context + gamma * I_ctx  # (B, L_c, D_in)
    V = X_tilde @ params["W_v"]                                  # (B, L_c, D_val)

    # Attention weights over CONTEXT positions for each QUERY position
    scores = jnp.einsum("bqd,bkd->bqk", Q, K) / jnp.sqrt(D_attn)  # (B, L_q, L_c)
    if linear:
        attn_weights = scores
    else:
        attn_weights = nn.softmax(scores, axis=-1)

    # Weighted sum of VALUES -> per-query outputs
    out = jnp.einsum("bqk,bkd->bqd", attn_weights, V)  # (B, L_q, D_val)
    return out, attn_weights


# Linear classifier
def init_classifier_params(key, val_dim, seq_len, out_dim=1, scale=0.02):
    k1, k2 = random.split(key)
    W_out = scale * random.normal(k1, (seq_len * val_dim, 1))
    b_out = jnp.zeros((out_dim,))
    return {"W_out": W_out, "b_out": b_out}


def forward_logits(params_attn, params_clf, X, gamma, linear=False):
    """
    X: (B, L, D_val)
    returns logits: (B,)
    """
    out, attn_w = gamma_value_self_attention_forward(
        params_attn, X, gamma=gamma, linear=linear,
    )  # out: (B, L, D_val)

    out_flat = out.reshape(out.shape[0], -1)  # (B, L * D_val)
    bias = jnp.broadcast_to(params_clf["b_out"], (out_flat.shape[0],))  # (B,)
    logits = (out_flat @ params_clf["W_out"]).squeeze(-1) + bias
    return logits, attn_w


def forward_multinomial_logits(params_attn, params_clf, X, gamma, linear=False):
    """
    X: (B, L, D_val)
    returns logits: (B,)
    """
    out, attn_w = gamma_value_self_attention_forward(
        params_attn, X, gamma=gamma, linear=linear,
    )  # out: (B, L, D_val)

    out_flat = out.reshape(out.shape[0], -1)  # (B, L * D_val)
    bias_shape = params_clf["b_out"].shape[-1]
    bias = jnp.broadcast_to(params_clf["b_out"][None,:], (out_flat.shape[0],bias_shape,))  # (B,)
    logits = (out_flat @ params_clf["W_out"]).squeeze(-1) + bias
    return logits, attn_w
