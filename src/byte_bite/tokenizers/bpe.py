import json
from typing import List, Dict, Tuple, Union
from collections import Counter
from ..base import BaseTokenizer

class BPETokenizer(BaseTokenizer):
    """Byte-level Byte Pair Encoding tokenizer"""

    def __init__(self):
        # Base vocabulary: all possible bytes (0-255)
        self.byte_to_token = {i: i for i in range(256)}
        self.token_to_byte = {i: bytes([i]) for i in range(256)}

        # Merge rules: stores the order of merges
        self.merges = {}  # (byte_pair) -> new_token_id
        self.vocab_size = 256  # Start with 256 base bytes

    def train(self, texts: Union[str, List[str]], vocab_size: int):
        """Train BPE on texts"""
        # Handle single string
        if isinstance(texts, str):
            texts = [texts]
        
        # Convert all texts to bytes
        corpus = []
        for text in texts:
            tokens = list(text.encode('utf-8'))  # Convert to list of bytes
            corpus.append(tokens)
        
        # Merge until we reach vocab_size
        num_merges = vocab_size - 256  # We start with 256 base bytes
        
        for i in range(num_merges):
            # Count all adjacent pairs across corpus
            pairs = self._count_pairs(corpus)
            
            if not pairs:
                break  # No more pairs to merge
            
            # Find most frequent pair
            best_pair = max(pairs, key=pairs.get)
            
            # Create new token for this pair
            new_token_id = 256 + i
            
            # Store the merge rule
            self.merges[best_pair] = new_token_id
            
            # Update vocab mappings
            self.token_to_byte[new_token_id] = (
                self.token_to_byte[best_pair[0]] + 
                self.token_to_byte[best_pair[1]]
            )
            
            # Apply merge to corpus
            corpus = self._apply_merge(corpus, best_pair, new_token_id)
        
        self.vocab_size = 256 + len(self.merges)

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs"""
        # Convert text to bytes
        tokens = list(text.encode('utf-8'))
        
        # Apply each merge rule in order
        # Wrap in list to reuse _apply_merge, then unwrap
        for pair, new_token_id in self.merges.items():
            tokens = self._apply_merge([tokens], pair, new_token_id)[0]
        
        return tokens
    
    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs back to text"""
        # Convert each token to its bytes
        byte_sequence = b''
        for token in tokens:
            byte_sequence += self.token_to_byte[token]
        
        # Decode bytes to string
        return byte_sequence.decode('utf-8', errors='replace')
    
    import json

    def save(self, path: str):
        """Save tokenizer to JSON file"""
        # Convert data to JSON-serializable format
        # Tuples → strings, bytes → lists
        merges_serializable = {
            f"{k[0]},{k[1]}": v for k, v in self.merges.items()
        }
        
        token_to_byte_serializable = {
            str(k): list(v) for k, v in self.token_to_byte.items()
        }
        
        data = {
            'merges': merges_serializable,
            'token_to_byte': token_to_byte_serializable,
            'vocab_size': self.vocab_size
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Tokenizer saved to {path}")

    def load(self, path: str):
        """Load tokenizer from JSON file"""
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Convert back from JSON format
        # Strings → tuples, lists → bytes
        self.merges = {
            tuple(map(int, k.split(','))): v 
            for k, v in data['merges'].items()
        }
        
        self.token_to_byte = {
            int(k): bytes(v) for k, v in data['token_to_byte'].items()
        }
        
        self.vocab_size = data['vocab_size']
        
        # Rebuild byte_to_token (not strictly necessary but good to have)
        self.byte_to_token = {i: i for i in range(256)}
        for token_id in range(256, self.vocab_size):
            if token_id in self.token_to_byte:
                self.byte_to_token[token_id] = token_id
        
        print(f"Tokenizer loaded from {path}")

    ##### HELPER METHODS #####

    def _count_pairs(self, corpus: List[List[int]]) -> Dict[Tuple[int, int], int]:
        """Count frequency of all adjacent token pairs in corpus"""
        pairs = Counter()
        
        for tokens in corpus:
            # Count pairs in this sequence
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                pairs[pair] += 1
        
        return pairs

    def _apply_merge(self, corpus: List[List[int]], pair: Tuple[int, int], 
                    new_token_id: int) -> List[List[int]]:
        """Apply a merge rule to the entire corpus"""
        new_corpus = []
        
        for tokens in corpus:
            new_tokens = []
            i = 0
            
            while i < len(tokens):
                # Check if we can merge at this position
                if (i < len(tokens) - 1 and 
                    tokens[i] == pair[0] and 
                    tokens[i + 1] == pair[1]):
                    # Merge the pair
                    new_tokens.append(new_token_id)
                    i += 2  # Skip both tokens
                else:
                    # Keep original token
                    new_tokens.append(tokens[i])
                    i += 1
            
            new_corpus.append(new_tokens)
        
        return new_corpus