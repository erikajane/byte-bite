import pytest
import os
from byte_bite import BPETokenizer

class TestBPETokenizer:
    
    def test_initialization(self):
        """Test tokenizer initializes correctly"""
        tokenizer = BPETokenizer()
        assert tokenizer.vocab_size == 256
        assert len(tokenizer.merges) == 0
    
    def test_train(self):
        """Test basic training"""
        tokenizer = BPETokenizer()
        texts = ["hello world", "hello there"]
        tokenizer.train(texts, vocab_size=300)
        
        assert tokenizer.vocab_size > 256
        assert len(tokenizer.merges) > 0
    
    def test_encode_decode(self):
        """Test encoding and decoding round trip"""
        tokenizer = BPETokenizer()
        tokenizer.train(["hello world"], vocab_size=300)
        
        text = "hello world"
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        
        assert text == decoded
        assert isinstance(encoded, list)
        assert all(isinstance(token, int) for token in encoded)
    
    def test_empty_text(self):
        """Test encoding empty string"""
        tokenizer = BPETokenizer()
        tokenizer.train(["test"], vocab_size=300)
        
        encoded = tokenizer.encode("")
        assert encoded == []
        
        decoded = tokenizer.decode([])
        assert decoded == ""
    
    def test_single_string_train(self):
        """Test training with single string instead of list"""
        tokenizer = BPETokenizer()
        tokenizer.train("hello world", vocab_size=300)
        
        assert tokenizer.vocab_size > 256
    
    def test_save_and_load(self):
        """Test saving and loading tokenizer"""
        # Train original
        tokenizer1 = BPETokenizer()
        tokenizer1.train(["hello world"], vocab_size=300)
        
        # Save
        test_path = "test_tokenizer.json"
        tokenizer1.save(test_path)
        
        # Load into new tokenizer
        tokenizer2 = BPETokenizer()
        tokenizer2.load(test_path)
        
        # Test they produce same results
        text = "hello world"
        assert tokenizer1.encode(text) == tokenizer2.encode(text)
        
        # Cleanup
        os.remove(test_path)
    
    def test_unseen_text(self):
        """Test encoding text not in training corpus"""
        tokenizer = BPETokenizer()
        tokenizer.train(["hello"], vocab_size=300)
        
        # This should still work, just with different tokens
        encoded = tokenizer.encode("world")
        decoded = tokenizer.decode(encoded)
        
        assert decoded == "world"