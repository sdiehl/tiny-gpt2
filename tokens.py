"""
Example usage of the tokenizer.
"""

from tinygpt2.encoder import get_encoder


def main():
    encoder = get_encoder("", "model")

    while True:
        try:
            text = input("Enter text to encode: ")
            encoded = encoder.encode(text)
            print(f"Tokens: {encoded}")
        except EOFError:
            print("\nDone.")
            break


if __name__ == "__main__":
    main()
