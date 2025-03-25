"""
GPT-2 model implementation using NumPy.

This module implements the core GPT-2 model architecture and text generation
functionality using the tensor operations and weight loading utilities.
"""

import numpy as np
from encoder import get_encoder
from gpt2_tensors import GPT2TensorManager
import gpt2_ops as ops
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def gpt2(inputs, wte, wpe, blocks, ln_f, n_head):
    """
    Forward pass through the GPT-2 model.

    Args:
        inputs: List of token IDs
        wte: Token embedding matrix
        wpe: Position embedding matrix
        blocks: List of transformer block parameters
        ln_f: Final layer normalization parameters
        n_head: Number of attention heads

    Returns:
        Logits for next token prediction
    """
    # Get token embeddings and position embeddings
    x = wte[inputs] + wpe[range(len(inputs))]

    # Apply transformer blocks
    for block in blocks:
        x = ops.transformer_block(x, **block, n_head=n_head)

    # Apply final layer norm and project to vocabulary
    x = ops.layer_norm(x, **ln_f)
    logits = x @ wte.T  # Project to vocabulary

    return logits


def generate(inputs, params, n_head, n_tokens_to_generate):
    """
    Generate tokens using the GPT-2 model.

    Args:
        inputs: Initial sequence of token IDs
        params: Model parameters dictionary
        n_head: Number of attention heads
        n_tokens_to_generate: Number of new tokens to generate

    Returns:
        List of generated token IDs
    """
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
    return inputs[len(inputs) - n_tokens_to_generate :]


def main(prompt: str, n_tokens_to_generate: int = 40):
    """
    Main entry point for text generation with GPT-2.

    Args:
        prompt: Input text to continue from
        n_tokens_to_generate: Number of new tokens to generate

    Returns:
        Generated text continuation
    """
    # Load model weights and configuration
    tensor_manager = GPT2TensorManager()
    params, hparams = tensor_manager.load_model_weights()

    # Load tokenizer
    encoder = get_encoder("", ".")

    # Encode input
    input_ids = encoder.encode(prompt)

    # Ensure we don't exceed context length
    if len(input_ids) + n_tokens_to_generate >= hparams["n_ctx"]:
        logger.warning(
            f"Input length + tokens to generate ({len(input_ids) + n_tokens_to_generate}) exceeds model context length ({hparams['n_ctx']})"
        )
        n_tokens_to_generate = hparams["n_ctx"] - len(input_ids) - 1
        logger.warning(f"Reducing tokens to generate to {n_tokens_to_generate}")

    # Generate tokens
    logger.info(f"Generating {n_tokens_to_generate} tokens for prompt: {prompt}")
    output_ids = generate(input_ids, params, hparams["n_head"], n_tokens_to_generate)

    # Decode the output
    output_text = encoder.decode(output_ids)
    return output_text


if __name__ == "__main__":
    # print(main("The rain in Spain falls mainly on the", 20))
    print(main("Stephen Hawking is a", 5))