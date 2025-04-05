"""
GPT-2 visualization tools using matplotlib to display attention patterns
during model inference.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
from encoder import get_encoder
from gpt2_tensors import ModelParams, HParams
import gpt2_ops as ops


def compute_attention_scores(
    inputs: List[int], params: ModelParams, n_head: int, block_idx: int = 0
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Compute attention scores for a specific transformer block.

    Args:
        inputs: List of token IDs
        params: Model parameters
        n_head: Number of attention heads
        block_idx: Index of the transformer block to visualize (default 0)

    Returns:
        raw_scores: Attention scores before masking
        masked_scores: Attention scores after masking
        tokens_text: List of token text representations
    """
    # Get token embeddings and position embeddings
    x = params.wte[inputs] + params.wpe[range(len(inputs))]

    # Apply layer norm
    block = params.blocks[block_idx]
    ln1_output = ops.layer_norm(x, g=block.ln_1.g, b=block.ln_1.b)

    # Get QKV projections
    qkv_proj = ln1_output @ block.attn.c_attn.w + block.attn.c_attn.b
    q_proj, k_proj, v_proj = np.split(qkv_proj, 3, axis=-1)

    # Reshape for attention computation
    seq_len = x.shape[0]
    head_size = x.shape[1] // n_head

    # Only compute for the first head for simplicity
    head_idx = 0
    q_head = q_proj[:, head_idx * head_size : (head_idx + 1) * head_size]
    k_head = k_proj[:, head_idx * head_size : (head_idx + 1) * head_size]

    # Compute raw attention scores
    raw_scores = (q_head @ k_head.T) / np.sqrt(head_size)

    # Create attention mask
    causal_mask = np.triu(np.ones((seq_len, seq_len)) * -1e10, k=1)

    # Apply mask
    masked_scores = raw_scores + causal_mask

    # Get token text representations
    encoder = get_encoder("", "model")
    tokens_text = [encoder.decode([token_id]) for token_id in inputs]

    return raw_scores, masked_scores, tokens_text


def display_qk_heatmap(attention_matrix: np.ndarray, tokens: List[str], title: str):
    """
    Display a heatmap of attention scores.

    Args:
        attention_matrix: Matrix of attention scores
        tokens: List of token text representations
        title: Title for the plot
    """
    plt.figure(figsize=(10, 8))
    im = plt.imshow(attention_matrix, cmap="viridis")

    # Set ticks and labels
    plt.xticks(range(len(tokens)), tokens, rotation=90)
    plt.yticks(range(len(tokens)), tokens)

    # Add colorbar and title
    plt.colorbar(im)
    plt.title(title)
    plt.tight_layout()


def visualize_attention(
    prompt: str,
    params: ModelParams,
    hparams: HParams,
    num_blocks: int = 3,
    save_path: Optional[str] = None,
):
    """
    Visualize attention patterns for multiple transformer blocks.

    Args:
        prompt: Input text prompt
        params: Model parameters
        hparams: Model hyperparameters
        num_blocks: Number of transformer blocks to visualize (default 3)
        save_path: Path to save the visualization (if None, display instead)
    """
    # Encode input
    encoder = get_encoder("", "model")
    input_ids = encoder.encode(prompt)

    # Ensure we don't exceed context length
    if len(input_ids) >= hparams.n_ctx:
        input_ids = input_ids[: hparams.n_ctx - 1]

    # Determine number of blocks to visualize
    num_blocks_to_show = min(num_blocks, len(params.blocks))

    # Create a figure with a grid based on the number of blocks
    fig, axes = plt.subplots(
        num_blocks_to_show, 3, figsize=(24, 6 * num_blocks_to_show)
    )

    # Ensure axes is 2D even if there's only one block
    if num_blocks_to_show == 1:
        axes = np.array([axes])

    # Compute and visualize attention scores for each block
    for i in range(num_blocks_to_show):
        raw_scores, masked_scores, tokens = compute_attention_scores(
            input_ids, params, hparams.n_head, i
        )

        # Plot raw attention scores in first column
        im1 = axes[i, 0].imshow(raw_scores, cmap="viridis")
        axes[i, 0].set_title(f"Raw QK Attention (Block {i})")
        axes[i, 0].set_xticks(range(len(tokens)))
        axes[i, 0].set_yticks(range(len(tokens)))
        axes[i, 0].set_xticklabels(tokens, rotation=90)
        axes[i, 0].set_yticklabels(tokens)
        fig.colorbar(im1, ax=axes[i, 0])

        # Plot masked attention scores in second column - use a different colormap to highlight masked areas
        masked_cmap = plt.cm.viridis.copy()  # type: ignore
        masked_cmap.set_under("black")  # Set color for values below the minimum

        # Create a mask to show exactly where the causal mask is applied
        vmin = np.min(raw_scores)  # Use minimum of raw scores
        im2 = axes[i, 1].imshow(masked_scores, cmap=masked_cmap, vmin=vmin)
        axes[i, 1].set_title(f"Masked Attention (Block {i})")
        axes[i, 1].set_xticks(range(len(tokens)))
        axes[i, 1].set_yticks(range(len(tokens)))
        axes[i, 1].set_xticklabels(tokens, rotation=90)
        axes[i, 1].set_yticklabels(tokens)
        fig.colorbar(im2, ax=axes[i, 1], extend="min")

        # Plot softmax probabilities in third column
        softmax_attn = np.exp(masked_scores) / np.sum(
            np.exp(masked_scores), axis=1, keepdims=True
        )
        im3 = axes[i, 2].imshow(
            softmax_attn, cmap="plasma"
        )  # Different colormap for contrast
        axes[i, 2].set_title(f"Attention Probabilities (Block {i})")
        axes[i, 2].set_xticks(range(len(tokens)))
        axes[i, 2].set_yticks(range(len(tokens)))
        axes[i, 2].set_xticklabels(tokens, rotation=90)
        axes[i, 2].set_yticklabels(tokens)
        fig.colorbar(im3, ax=axes[i, 2])

    plt.suptitle(f"Attention Visualization for: '{prompt}'", fontsize=16)
    plt.tight_layout(rect=(0, 0, 1, 0.97))  # Make room for the suptitle

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


if __name__ == "__main__":
    from gpt2_tensors import load_gpt2_weights

    # Load model weights and configuration
    print("Loading model weights...")
    params, hparams = load_gpt2_weights()
    print("Model loaded successfully")

    # Example prompt
    prompt = "The quick brown fox jumps over the lazy dog"

    # Visualize attention for the first three transformer blocks
    visualize_attention(prompt, params, hparams, num_blocks=1)
