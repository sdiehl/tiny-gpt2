"""
GPT-2 model implementation using NumPy.

This module implements the core GPT-2 model architecture and text generation
functionality using the tensor operations and weight loading utilities.
"""

import numpy as np
from encoder import get_encoder
from gpt2_tensors import load_gpt2_weights, ModelParams, HParams
import gpt2_ops as ops
import logging
import readline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def gpt2(inputs: list[int], params: ModelParams, n_head: int) -> np.ndarray:
    """
    Forward pass through the GPT-2 model.

    Args:
        inputs: List of token IDs
        params: Model parameters dictionary
        n_head: Number of attention heads

    Returns:
        Logits for next token prediction
    """
    # Get token embeddings and position embeddings
    x = params.wte[inputs] + params.wpe[range(len(inputs))]

    # Apply transformer block stack
    x = ops.transformer_block(
        x,
        n_head=n_head,
        mlp=params.blocks[0].mlp,
        attn=params.blocks[0].attn,
        ln_1=params.blocks[0].ln_1,
        ln_2=params.blocks[0].ln_2,
    )
    x = ops.transformer_block(
        x,
        n_head=n_head,
        mlp=params.blocks[1].mlp,
        attn=params.blocks[1].attn,
        ln_1=params.blocks[1].ln_1,
        ln_2=params.blocks[1].ln_2,
    )
    x = ops.transformer_block(
        x,
        n_head=n_head,
        mlp=params.blocks[2].mlp,
        attn=params.blocks[2].attn,
        ln_1=params.blocks[2].ln_1,
        ln_2=params.blocks[2].ln_2,
    )
    x = ops.transformer_block(
        x,
        n_head=n_head,
        mlp=params.blocks[3].mlp,
        attn=params.blocks[3].attn,
        ln_1=params.blocks[3].ln_1,
        ln_2=params.blocks[3].ln_2,
    )
    x = ops.transformer_block(
        x,
        n_head=n_head,
        mlp=params.blocks[4].mlp,
        attn=params.blocks[4].attn,
        ln_1=params.blocks[4].ln_1,
        ln_2=params.blocks[4].ln_2,
    )
    x = ops.transformer_block(
        x,
        n_head=n_head,
        mlp=params.blocks[5].mlp,
        attn=params.blocks[5].attn,
        ln_1=params.blocks[5].ln_1,
        ln_2=params.blocks[5].ln_2,
    )
    x = ops.transformer_block(
        x,
        n_head=n_head,
        mlp=params.blocks[6].mlp,
        attn=params.blocks[6].attn,
        ln_1=params.blocks[6].ln_1,
        ln_2=params.blocks[6].ln_2,
    )
    x = ops.transformer_block(
        x,
        n_head=n_head,
        mlp=params.blocks[7].mlp,
        attn=params.blocks[7].attn,
        ln_1=params.blocks[7].ln_1,
        ln_2=params.blocks[7].ln_2,
    )
    x = ops.transformer_block(
        x,
        n_head=n_head,
        mlp=params.blocks[8].mlp,
        attn=params.blocks[8].attn,
        ln_1=params.blocks[8].ln_1,
        ln_2=params.blocks[8].ln_2,
    )
    x = ops.transformer_block(
        x,
        n_head=n_head,
        mlp=params.blocks[9].mlp,
        attn=params.blocks[9].attn,
        ln_1=params.blocks[9].ln_1,
        ln_2=params.blocks[9].ln_2,
    )
    x = ops.transformer_block(
        x,
        n_head=n_head,
        mlp=params.blocks[10].mlp,
        attn=params.blocks[10].attn,
        ln_1=params.blocks[10].ln_1,
        ln_2=params.blocks[10].ln_2,
    )
    x = ops.transformer_block(
        x,
        n_head=n_head,
        mlp=params.blocks[11].mlp,
        attn=params.blocks[11].attn,
        ln_1=params.blocks[11].ln_1,
        ln_2=params.blocks[11].ln_2,
    )

    # Apply final layer norm and project to vocabulary
    x = ops.layer_norm(x, g=params.ln_f.g, b=params.ln_f.b)
    logits = x @ params.wte.T  # Project to vocabulary

    return logits


def generate(
    inputs: list[int],
    params: ModelParams,
    n_head: int,
    n_tokens_to_generate: int,
    encoder=None,
) -> list[int]:
    inputs = list(inputs)  # Make a copy to avoid modifying the original
    generated_tokens = []

    for i in range(n_tokens_to_generate):
        # Get logits for the entire sequence
        logits = gpt2(inputs, params, n_head=n_head)

        # Get the next token ID from the last position
        next_id = np.argmax(logits[-1])
        next_id_int = int(next_id)

        # Add the predicted token to the sequence
        inputs.append(next_id_int)
        generated_tokens.append(next_id_int)

        # Stream the token if encoder is provided
        if encoder:
            token_text = encoder.decode([next_id_int])
            print(token_text, end="", flush=True)

    return generated_tokens


def run(
    params: ModelParams, hparams: HParams, prompt: str, n_tokens_to_generate: int = 40
) -> str:
    # Load tokenizer
    encoder = get_encoder("", "model")

    # Encode input
    input_ids = encoder.encode(prompt)

    # Ensure we don't exceed context length
    if len(input_ids) + n_tokens_to_generate >= hparams.n_ctx:
        logger.warning(
            f"Input length + tokens to generate ({len(input_ids) + n_tokens_to_generate}) exceeds model context length ({hparams.n_ctx})"
        )
        n_tokens_to_generate = hparams.n_ctx - len(input_ids) - 1
        logger.warning(f"Reducing tokens to generate to {n_tokens_to_generate}")

    # Print the initial prompt
    print(prompt, end="", flush=True)

    # Generate tokens with streaming
    output_ids = generate(
        input_ids, params, hparams.n_head, n_tokens_to_generate, encoder=encoder
    )
    print()  # Add newline after generation

    # Return the generated text (though it's already been printed)
    return encoder.decode(output_ids)


def main():
    # Load model weights and configuration
    print("Loading model weights...")
    params, hparams = load_gpt2_weights()
    print("Model loaded successfully")

    # Configure readline
    readline.parse_and_bind("tab: complete")
    readline.set_history_length(100)

    while True:
        try:
            # Get user input with readline support
            prompt = input("Enter a prompt: ")
            run(params, hparams, prompt.strip())
        except KeyboardInterrupt:
            print("\nInterrupted by user")
            continue
        except EOFError:
            print("\nDone.")
            break


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    main()
    # print(main("The rain in Spain falls mainly in the", 40))
    # print(main("You're a wizard,", 40))
    # print(main("What is the capital of France?", 10))
    # print(main("Stephen Hawking is a", 40))
    # print(main("The quick brown fox jumped", 10))
    # print(main("Star Wars is a movie about", 40))

    # This is a known good prompt
    # print(main("Alan Turing theorized that computers would one day become", 10))
    # ... the most powerful machines on the planet.
