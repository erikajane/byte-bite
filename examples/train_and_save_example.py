from src.byte_bite import BPETokenizer

# Train
texts = ["hello world", "hello there", "world peace"]
tokenizer = BPETokenizer()
print("Training...")
tokenizer.train(texts, vocab_size=300)

# Save
tokenizer.save("my_tokenizer.json")

# Test original
test_text = "hello world"
encoded1 = tokenizer.encode(test_text)
print(f"\nOriginal tokenizer encoded: {encoded1}")

# Load into new tokenizer
tokenizer2 = BPETokenizer()
tokenizer2.load("my_tokenizer.json")

# Test loaded
encoded2 = tokenizer2.encode(test_text)
print(f"Loaded tokenizer encoded: {encoded2}")
print(f"Match: {encoded1 == encoded2}")

decoded = tokenizer2.decode(encoded2)
print(f"Decoded: '{decoded}'")