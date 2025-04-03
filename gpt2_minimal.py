import numpy as np
from gpt2_tensors import LayerNormParams, MLPParams, AttentionParams


# The entire transformer block inlined
def transformer_block(
    x: np.ndarray,
    mlp: MLPParams,
    attn: AttentionParams,
    ln_1: LayerNormParams,
    ln_2: LayerNormParams,
    n_head: int,
) -> np.ndarray:
    embedding_dim = x.shape[-1]
    seq_len = x.shape[0]
    head_size = embedding_dim // n_head

    ln1_mean = np.mean(x, axis=-1, keepdims=True)
    ln1_variance = np.var(x, axis=-1, keepdims=True)
    ln1_normalized = (x - ln1_mean) / np.sqrt(ln1_variance + 1e-5)
    ln1_output = ln_1.g * ln1_normalized + ln_1.b

    qkv_proj = ln1_output @ attn.c_attn.w + attn.c_attn.b
    q_proj, k_proj, v_proj = np.split(qkv_proj, 3, axis=-1)

    q_heads = q_proj.reshape(seq_len, n_head, head_size)
    k_heads = k_proj.reshape(seq_len, n_head, head_size)
    v_heads = v_proj.reshape(seq_len, n_head, head_size)

    causal_mask = np.triu(np.ones((seq_len, seq_len), dtype=x.dtype) * -np.inf, k=1)

    q_heads_t = q_heads.transpose(1, 0, 2)
    k_heads_t = k_heads.transpose(1, 0, 2)
    v_heads_t = v_heads.transpose(1, 0, 2)

    attention_scores = (q_heads_t @ k_heads_t.transpose(0, 2, 1)) / np.sqrt(head_size)
    attention_scores = attention_scores + causal_mask

    exp_scores = np.exp(
        attention_scores - np.max(attention_scores, axis=-1, keepdims=True)
    )
    attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    weighted_values = attention_weights @ v_heads_t
    merged_heads = weighted_values.transpose(1, 0, 2).reshape(seq_len, embedding_dim)
    mha_output = merged_heads @ attn.c_proj.w + attn.c_proj.b

    x = x + mha_output

    ln2_mean = np.mean(x, axis=-1, keepdims=True)
    ln2_variance = np.var(x, axis=-1, keepdims=True)
    ln2_normalized = (x - ln2_mean) / np.sqrt(ln2_variance + 1e-5)
    ln2_output = ln_2.g * ln2_normalized + ln_2.b

    fc_output = ln2_output @ mlp.c_fc.w + mlp.c_fc.b
    gelu_output = (
        0.5
        * fc_output
        * (1 + np.tanh(np.sqrt(2 / np.pi) * (fc_output + 0.044715 * fc_output**3)))
    )
    ffn_output = gelu_output @ mlp.c_proj.w + mlp.c_proj.b

    x = x + ffn_output

    return x
