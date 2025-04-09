"""
Example usage of the tokenizer.
"""

from tinygpt2 import get_encoder


def main():
    encoder = get_encoder("", "model")

    while True:
        try:
            text = input("Enter text to encode: ")
            encoded_ids = encoder.encode(text)

            # Decode each token ID back to its string representation
            decoded_tokens = [encoder.decode([token_id]) for token_id in encoded_ids]

            print("Token Mapping:")
            for token, token_id in zip(decoded_tokens, encoded_ids):
                print(f"  '{token}': {token_id}")

        except EOFError:
            print("\nDone.")
            break


if __name__ == "__main__":
    main()
