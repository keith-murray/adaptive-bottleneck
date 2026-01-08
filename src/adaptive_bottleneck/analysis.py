import numpy as np
import matplotlib.pyplot as plt


def compute_scores_matrix(
    params_attn,
):
    # --- Extract parameters ---
    Wq = np.array(params_attn["W_q"])  # (D_in, D_attn)
    Wk = np.array(params_attn["W_k"])  # (D_in, D_attn)

    D_in, D_attn = Wq.shape
    L = D_in  # by design

    # --- Identity probe ---
    X = np.eye(L, dtype=np.float32)[None, :, :]  # (1, L, D_in)

    Q = X @ Wq  # (1, L, D_attn)
    K = X @ Wk  # (1, L, D_attn)

    scores = (Q @ np.transpose(K, (0, 2, 1))) / np.sqrt(D_attn)  # (1, L, L)
    S0 = scores[0]

    return S0

def plot_attention_representations(
    params_attn,
    suptitle="Learned attention representations",
    save_path=None,
):
    """
    Plot W_q, W_k, pre-softmax attention scores, and W_v in a 2x2 panel.

    Panels:
        (0,0): W_q
        (0,1): W_k
        (1,0): Scores = Q K^T / sqrt(d)
        (1,1): W_v

    Args:
        params_attn (dict): Attention parameters with keys "W_q", "W_k", "W_v".
        suptitle (str): Figure suptitle.
        save_path (str or None): If provided, saves the figure to this path.
    """
    # --- Extract parameters ---
    Wq = np.array(params_attn["W_q"])  # (D_in, D_attn)
    Wk = np.array(params_attn["W_k"])  # (D_in, D_attn)
    Wv = np.array(params_attn["W_v"])  # (D_in, D_val)

    S0 = compute_scores_matrix(params_attn)

    # --- Plot ---
    fig, axes = plt.subplots(2, 2, figsize=(8, 7), constrained_layout=True)

    # W_q
    im = axes[0, 0].imshow(Wq, aspect="auto")
    axes[0, 0].set_title("W_q")
    axes[0, 0].set_xlabel("attention dim")
    axes[0, 0].set_ylabel("input dim")
    plt.colorbar(im, ax=axes[0, 0], fraction=0.046, pad=0.04)

    # W_k
    im = axes[0, 1].imshow(Wk, aspect="auto")
    axes[0, 1].set_title("W_k")
    axes[0, 1].set_xlabel("attention dim")
    axes[0, 1].set_ylabel("input dim")
    plt.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04)

    # Scores
    im = axes[1, 0].imshow(S0, aspect="auto")
    axes[1, 0].set_title(r"Scores $= QK^\top / \sqrt{d}$ (pre-softmax)")
    axes[1, 0].set_xlabel("key index")
    axes[1, 0].set_ylabel("query index")
    plt.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04)

    # W_v
    im = axes[1, 1].imshow(Wv, aspect="auto")
    axes[1, 1].set_title("W_v")
    axes[1, 1].set_xlabel("value dim")
    axes[1, 1].set_ylabel("input dim")
    plt.colorbar(im, ax=axes[1, 1], fraction=0.046, pad=0.04)

    fig.suptitle(suptitle, fontsize=12)

    if save_path is not None:
        plt.savefig(save_path, dpi=300)

    plt.show()
