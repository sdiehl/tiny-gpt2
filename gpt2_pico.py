import numpy as np
from encoder import get_encoder
from gpt2_loader import GPT2WeightLoader, download_vocab_encoder
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def layer_norm(x, g, b, eps: float = 1e-5):
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    return g * (x - mean) / np.sqrt(variance + eps) + b

def linear(x, w, b):
    return x @ w + b

def ffn(x, c_fc, c_proj):
    return linear(gelu(linear(x, **c_fc)), **c_proj)

def attention(q, k, v, mask):
    # Scaled dot-product attention with causal mask
    # q, k, v shape: [seq_len, head_dim]
    # mask shape: [seq_len, seq_len]
    attention_scores = (q @ k.T) / np.sqrt(q.shape[-1])
    attention_scores = attention_scores + mask
    attention_weights = softmax(attention_scores)
    return attention_weights @ v

def mha(x, c_attn, c_proj, n_head):
    # Multi-head attention
    # x shape: [seq_len, n_embd]
    # Split into qkv
    x_proj = linear(x, **c_attn)  # [seq_len, 3*n_embd]
    
    # Split into q, k, v and then into heads
    qkv = np.split(x_proj, 3, axis=-1)  # 3 tensors of [seq_len, n_embd]
    n_embd = qkv[0].shape[-1]
    head_dim = n_embd // n_head
    
    # Reshape into heads
    qkv_heads = []
    for t in qkv:
        # Reshape to [seq_len, n_head, head_dim]
        reshaped = t.reshape(t.shape[0], n_head, head_dim)
        # Transpose to [n_head, seq_len, head_dim]
        transposed = np.transpose(reshaped, (1, 0, 2))
        qkv_heads.append(transposed)
    
    # Causal mask to prevent attending to future tokens
    seq_len = x.shape[0]
    causal_mask = (1 - np.tri(seq_len, dtype=x.dtype)) * -1e10
    
    # Apply attention for each head
    out_heads = []
    for h in range(n_head):
        q, k, v = qkv_heads[0][h], qkv_heads[1][h], qkv_heads[2][h]
        out_heads.append(attention(q, k, v, causal_mask))
    
    # Concatenate heads and project
    # Reshape from [n_head, seq_len, head_dim] to [seq_len, n_embd]
    out_concat = np.concatenate([h.reshape(seq_len, -1) for h in out_heads], axis=-1)
    
    # Final projection
    return linear(out_concat, **c_proj)

def transformer_block(x, mlp, attn, ln_1, ln_2, n_head):
    # Apply first residual connection with attention
    a = layer_norm(x, **ln_1)
    a = mha(a, **attn, n_head=n_head)
    x = x + a
    
    # Apply second residual connection with MLP
    m = layer_norm(x, **ln_2)
    m = ffn(m, **mlp)
    x = x + m
    
    return x

def gpt2(inputs, wte, wpe, blocks, ln_f, n_head):
    # Get token embeddings and position embeddings
    # inputs: list of token ids
    x = wte[inputs] + wpe[range(len(inputs))]
    
    # Apply transformer blocks
    for block in blocks:
        x = transformer_block(x, **block, n_head=n_head)
    
    # Apply final layer norm and project to vocabulary
    x = layer_norm(x, **ln_f)
    logits = x @ wte.T  # Project to vocabulary
    
    return logits

def generate(inputs, params, n_head, n_tokens_to_generate):
    """Generate tokens using the GPT-2 model"""
    # Use a simple progress indicator instead of tqdm to avoid linter errors
    inputs = list(inputs)  # Make a copy to avoid modifying the original
    
    for i in range(n_tokens_to_generate):
        # Print progress
        print(f"Generating token {i+1}/{n_tokens_to_generate}", end="\r")
        
        # Get logits for the entire sequence
        logits = gpt2(inputs, **params, n_head=n_head)
        
        # Get the next token ID from the last position
        next_id = np.argmax(logits[-1])
        
        # Add the predicted token to the sequence
        inputs.append(int(next_id))
    
    print()  # Newline after progress
    
    # Return only the newly generated tokens
    return inputs[len(inputs) - n_tokens_to_generate:]

def load_gpt2_weights():
    """Load and organize GPT2 weights into the format expected by our implementation."""
    # First download vocab and encoder files if needed
    download_vocab_encoder(".")
    
    # Initialize and download weights
    loader = GPT2WeightLoader("openai-community/gpt2")
    loader.download_weights()

    # Load config for hyperparameters
    hparams = loader.config
    logger.info(f"Model config: {hparams}")
    
    # Helper to get tensor by name
    def get_tensor(name):
        tensor = loader.get_tensor(name)
        logger.info(f"Loaded tensor {name} with shape {tensor.shape}")
        return tensor

    # Organize weights into our expected format
    wte = get_tensor("wte.weight")  # Token embeddings
    wpe = get_tensor("wpe.weight")  # Position embeddings
    
    # Final layer norm
    ln_f = {
        "g": get_tensor("ln_f.weight"),
        "b": get_tensor("ln_f.bias")
    }
    
    # Load transformer blocks
    blocks = []
    for i in range(hparams["n_layer"]):
        prefix = f"h.{i}"
        logger.info(f"\nLoading block {i}")
        
        # Get attention weights and handle their shapes
        c_attn_w = get_tensor(f"{prefix}.attn.c_attn.weight")  # [768, 2304]
        c_attn_b = get_tensor(f"{prefix}.attn.c_attn.bias")    # [2304]
        c_proj_w = get_tensor(f"{prefix}.attn.c_proj.weight")  # [768, 768]
        c_proj_b = get_tensor(f"{prefix}.attn.c_proj.bias")    # [768]
        
        # Get MLP weights
        c_fc_w = get_tensor(f"{prefix}.mlp.c_fc.weight")    # [768, 3072]
        c_fc_b = get_tensor(f"{prefix}.mlp.c_fc.bias")      # [3072]
        c_proj2_w = get_tensor(f"{prefix}.mlp.c_proj.weight")  # [3072, 768]
        c_proj2_b = get_tensor(f"{prefix}.mlp.c_proj.bias")    # [768]
        
        block = {
            "ln_1": {
                "g": get_tensor(f"{prefix}.ln_1.weight"),
                "b": get_tensor(f"{prefix}.ln_1.bias")
            },
            "ln_2": {
                "g": get_tensor(f"{prefix}.ln_2.weight"),
                "b": get_tensor(f"{prefix}.ln_2.bias")
            },
            "mlp": {
                "c_fc": {
                    "w": c_fc_w,  # [768, 3072] for x @ w -> [batch_size, 3072]
                    "b": c_fc_b   # [3072]
                },
                "c_proj": {
                    "w": c_proj2_w,  # [3072, 768] for x @ w -> [batch_size, 768]
                    "b": c_proj2_b     # [768]
                }
            },
            "attn": {
                "c_attn": {
                    "w": c_attn_w,  # [768, 2304] for x @ w -> [batch_size, 2304]
                    "b": c_attn_b   # [2304]
                },
                "c_proj": {
                    "w": c_proj_w,  # [768, 768] for x @ w -> [batch_size, 768]
                    "b": c_proj_b   # [768]
                }
            }
        }
        blocks.append(block)
    
    params = {
        "wte": wte,
        "wpe": wpe,
        "blocks": blocks,
        "ln_f": ln_f
    }
    
    return params, hparams

def main(prompt: str, n_tokens_to_generate: int = 40):
    # Load weights and hyperparameters
    params, hparams = load_gpt2_weights()
    
    # Load tokenizer
    encoder = get_encoder("", ".")
    
    # Encode input and generate
    input_ids = encoder.encode(prompt)
    
    # Ensure we don't exceed context length
    if len(input_ids) + n_tokens_to_generate >= hparams["n_ctx"]:
        logger.warning(f"Input length + tokens to generate ({len(input_ids) + n_tokens_to_generate}) exceeds model context length ({hparams['n_ctx']})")
        n_tokens_to_generate = hparams["n_ctx"] - len(input_ids) - 1
        logger.warning(f"Reducing tokens to generate to {n_tokens_to_generate}")
    
    # Generate tokens
    logger.info(f"Generating {n_tokens_to_generate} tokens for prompt: {prompt}")
    output_ids = generate(input_ids, params, hparams["n_head"], n_tokens_to_generate)
    
    # Decode the output
    output_text = encoder.decode(output_ids)
    return output_text

if __name__ == "__main__":
    # print(main("The rain in Spain falls mainly on the", 40))
    print(main("Stephen Hawking is a", 40))