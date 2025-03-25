import numpy as np
from gpt2_tensors import LayerNormParams, LinearParams, MLPParams, AttentionParams


def gelu(x: np.ndarray) -> np.ndarray:
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def softmax(x: np.ndarray) -> np.ndarray:
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def layer_norm(
    x: np.ndarray, g: np.ndarray, b: np.ndarray, eps: float = 1e-5
) -> np.ndarray:
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    return g * (x - mean) / np.sqrt(variance + eps) + b


def linear(x: np.ndarray, w: np.ndarray, b: np.ndarray) -> np.ndarray:
    return x @ w + b


def ffn(x: np.ndarray, c_fc: LinearParams, c_proj: LinearParams) -> np.ndarray:
    return linear(gelu(linear(x, w=c_fc.w, b=c_fc.b)), w=c_proj.w, b=c_proj.b)


# XXX: Expand c_fc and c_proj into their two array arguments for easier inlining
# def ffn(x: np.ndarray, c_fc_w: np.ndarray, c_fc_b: np.ndarray, c_proj_w: np.ndarray, c_proj_b: np.ndarray) -> np.ndarray:
#     return linear(gelu(linear(x, w=c_fc_w, b=c_fc_b)), w=c_proj_w, b=c_proj_b)


def attention(
    q: np.ndarray, k: np.ndarray, v: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    attention_scores = (q @ k.T) / np.sqrt(q.shape[-1])
    attention_scores = attention_scores + mask
    attention_weights = softmax(attention_scores)
    return attention_weights @ v


def mha(
    x: np.ndarray,
    c_attn: LinearParams,
    c_proj: LinearParams,
    n_head: int,
) -> np.ndarray:
    # Project input to Q, K, V
    x_proj = linear(x, w=c_attn.w, b=c_attn.b)

    # Split into q, k, v and reshape for multiple heads
    qkv = np.split(x_proj, 3, axis=-1)
    n_embd = qkv[0].shape[-1]
    head_dim = n_embd // n_head

    # Reshape and transpose for attention
    qkv_heads = []
    for t in qkv:
        reshaped = t.reshape(t.shape[0], n_head, head_dim)
        transposed = np.transpose(reshaped, (1, 0, 2))
        qkv_heads.append(transposed)

    # Causal mask prevents attending to future tokens
    seq_len = x.shape[0]
    causal_mask = (1 - np.tri(seq_len, dtype=x.dtype)) * -1e10

    # Apply attention for each head
    out_heads = []
    for h in range(n_head):
        q, k, v = qkv_heads[0][h], qkv_heads[1][h], qkv_heads[2][h]
        out_heads.append(attention(q, k, v, causal_mask))

    # Concatenate heads and project
    out_concat = np.concatenate([h.reshape(seq_len, -1) for h in out_heads], axis=-1)
    return linear(out_concat, w=c_proj.w, b=c_proj.b)


def transformer_block(
    x: np.ndarray,
    mlp: MLPParams,
    attn: AttentionParams,
    ln_1: LayerNormParams,
    ln_2: LayerNormParams,
    n_head: int,
) -> np.ndarray:
    # First sub-block: Layer norm -> Attention -> Residual
    a = layer_norm(x, g=ln_1.g, b=ln_1.b)
    a = mha(a, c_attn=attn.c_attn, c_proj=attn.c_proj, n_head=n_head)
    x = x + a

    # Second sub-block: Layer norm -> FFN -> Residual
    m = layer_norm(x, g=ln_2.g, b=ln_2.b)
    m = ffn(m, c_fc=mlp.c_fc, c_proj=mlp.c_proj)
    x = x + m

    return x
