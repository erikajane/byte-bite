# byte-bite 🔤

A simple, educational library for training your own tokenizers from scratch. Currently supports Byte-Level Byte Pair Encoding (BPE).

## Features

- **Byte-Level BPE**: Train tokenizers that handle any Unicode text
- **Save/Load**: Persist trained tokenizers for reuse
- **Clean API**: Simple, intuitive interface
- **Extensible**: Built with base classes to support multiple tokenization algorithms

## Installation

### From source (development)
```bash
git clone https://github.com/erikajane/byte-bite.git
cd byte-bite
python3 -m venv byte-bite_env
source byte-bite_env/bin/activate  # On Windows: byte-bite_env\Scripts\activate
pip install -e .
```

## Quick Start
```python
from byte_bite import BPETokenizer

# Create and train a tokenizer
tokenizer = BPETokenizer()
texts = [
    "hello world",
    "hello there", 
    "world peace"
]
tokenizer.train(texts, vocab_size=500)

# Encode text to token IDs
tokens = tokenizer.encode("hello world")
print(tokens)  # [265]

# Decode back to text
text = tokenizer.decode(tokens)
print(text)  # "hello world"

# Save for later use
tokenizer.save("my_tokenizer.json")

# Load saved tokenizer
new_tokenizer = BPETokenizer()
new_tokenizer.load("my_tokenizer.json")
```

## API Reference

### BPETokenizer

**Methods:**

- `train(texts: Union[str, List[str]], vocab_size: int)` - Train the tokenizer on a corpus
- `encode(text: str) -> List[int]` - Convert text to token IDs
- `decode(tokens: List[int]) -> str` - Convert token IDs back to text
- `save(path: str)` - Save tokenizer to JSON file
- `load(path: str)` - Load tokenizer from JSON file

## Development

### Running Tests
```bash
# Install dev dependencies
pip install pytest

# Run tests
pytest tests/ -v
```

### Project Structure
```
byte-bite/
├── src/
│   └── byte_bite/
│       ├── __init__.py
│       ├── base.py           # Base tokenizer interface
│       └── tokenizers/
│           ├── __init__.py
│           └── bpe.py        # BPE implementation
├── tests/
│   └── test_bpe.py
├── examples/
│   └── train_and_save_example.py
└── README.md
```

## How It Works

Byte-Level BPE works by:

1. **Starting with bytes**: Text is encoded as UTF-8 bytes (256 base tokens)
2. **Counting pairs**: Find the most frequent adjacent token pairs
3. **Merging**: Replace the most frequent pair with a new token
4. **Repeating**: Continue until reaching desired vocabulary size

This approach was popularized by GPT-2 and handles any Unicode text without unknown tokens.

## Roadmap

- [ ] Add WordPiece tokenizer
- [ ] Add Unigram tokenizer
- [ ] Pre-tokenization options (whitespace, punctuation)
- [ ] Special token support (`<pad>`, `<unk>`, `<bos>`, `<eos>`)
- [ ] Performance optimizations
- [ ] Publish to PyPI

## Contributing

Contributions welcome! Please feel free to submit a Pull Request.

## License

MIT License - see LICENSE file for details

## Acknowledgments

Inspired by modern tokenization libraries and built for educational purposes.