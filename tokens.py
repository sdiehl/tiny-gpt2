"""
Example usage of the tokenizer.
"""

from encoder import get_encoder


def main():
    encoder = get_encoder("", "model")

    while True:
        text = input("Enter text to encode: ")
        encoded = encoder.encode(text)
        print(f"Tokens: {encoded}")


if __name__ == "__main__":
    main()
