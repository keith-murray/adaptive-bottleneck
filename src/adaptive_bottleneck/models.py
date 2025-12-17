import jax
import jax.numpy as jnp


def init_gamma_value_attn_params(key, embed_dim, head_dim, max_seq_len, use_pos_proj=True, scale=0.02):
    """
    Initialize parameters for gamma-value self-attention in pure JAX.

    Shapes:
      W_q:    (embed_dim, head_dim)
      W_k:    (embed_dim, head_dim)
      W_v:    (embed_dim, head_dim)
      W_pos:  (max_seq_len, embed_dim)   # only used if embed_dim != seq_len and use_pos_proj=True
    """
    k1, k2, k3, k4 = jax.random.split(key, 4)

    params = {
        "W_q": scale * jax.random.normal(k1, (embed_dim, head_dim)),
        "W_k": scale * jax.random.normal(k2, (embed_dim, head_dim)),
        "W_v": scale * jax.random.normal(k3, (embed_dim, head_dim)),
    }
    if use_pos_proj:
        params["W_pos"] = scale * jax.random.normal(k4, (max_seq_len, embed_dim))

    return params


def softmax(x, axis=-1):
    x = x - jnp.max(x, axis=axis, keepdims=True)
    return jnp.exp(x) / jnp.sum(jnp.exp(x), axis=axis, keepdims=True)


def gamma_value_attention_forward(params, X, gamma, use_pos_proj=True):
    """
    Pure JAX forward pass.

    Args:
      params: dict from init_gamma_value_attn_params
      X:      (B, L, D_in) input sequence
      gamma:  scalar in [0, 1]
      linear: if True, skip softmax normalization
      use_pos_proj: if True, use params["W_pos"] when D_in != L

    Returns:
      out:         (B, L, D_h)
      attn_weights:(B, L, L)
    """
    B, L, D_in = X.shape
    D_h = params["W_q"].shape[1]

    # Projections
    Q = X @ params["W_q"]          # (B, L, D_h)
    K = X @ params["W_k"]          # (B, L, D_h)

    # Build I_embed with shape (B, L, D_in)
    if D_in == L:
        # literal identity basis in embedding space
        I_embed = jnp.broadcast_to(jnp.eye(L, dtype=X.dtype)[None, :, :], (B, L, D_in))
    else:
        if not use_pos_proj:
            raise ValueError("embed_dim != seq_len but use_pos_proj=False; cannot form I in embed space.")
        # one-hot positions projected to embed space: (L, D_in)
        I_embed_single = params["W_pos"][:L, :]   # (L, D_in)
        I_embed = jnp.broadcast_to(I_embed_single[None, :, :], (B, L, D_in))

    # Gamma-interpolated input to values
    X_tilde = (1.0 - gamma) * X + gamma * I_embed  # (B, L, D_in)

    # Values
    V = X_tilde @ params["W_v"]   # (B, L, D_h)

    # Attention weights
    scores = jnp.einsum("bld,bmd->blm", Q, K) / jnp.sqrt(D_h)   # (B, L, L)
    attn_weights = softmax(scores, axis=-1)

    # Output
    out = jnp.einsum("blm,bmd->bld", attn_weights, V)          # (B, L, D_h)
    return out, attn_weights