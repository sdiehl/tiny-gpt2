from encoder import get_encoder

encoder = get_encoder("", ".")
encoded = encoder.encode("Hello, world!")

print(encoded)
