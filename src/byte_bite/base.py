from abc import ABC, abstractmethod
from typing import List, Union

class BaseTokenizer:
    """Base class for all tokenizers"""
    
    @abstractmethod
    def train(self, texts, vocab_size):
        """Train the tokenizer on corpus"""
        pass
    @abstractmethod
    def encode(self, text):
        """Encode text to token IDs"""
        pass
    @abstractmethod
    def decode(self, tokens):
        """Decode token IDs back to text"""
        pass
    @abstractmethod
    def save(self, path): 
        """Save tokenizer to file"""
        pass
    @abstractmethod
    def load(self, path): 
        """Load tokenizer from file"""
        pass