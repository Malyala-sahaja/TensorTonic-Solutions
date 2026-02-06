import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        # YOUR CODE HERE
        special_tokens = [self.pad_token, self.unk_token, self.bos_token, self.eos_token]
        
        # Add special tokens first
        for idx, token in enumerate(special_tokens):
            self.word_to_id[token] = idx
            self.id_to_word[idx] = token
        
        # Collect unique words from all texts
        unique_words = set()
        for text in texts:
            words = text.split()
            unique_words.update(words)
        
        # Remove special tokens if they appear in the text
        unique_words = unique_words - set(special_tokens)
        
        # Sort for consistent ordering
        unique_words = sorted(unique_words)
        
        # Add unique words to vocabulary (starting from ID 4)
        current_id = len(special_tokens)
        for word in unique_words:
            self.word_to_id[word] = current_id
            self.id_to_word[current_id] = word
            current_id += 1
        
        # Update vocab size
        self.vocab_size = len(self.word_to_id)
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        # YOUR CODE HERE
        words = text.split()
        # Use .get() with UNK token ID as default for unknown words
        unk_id = self.word_to_id[self.unk_token]
        return [self.word_to_id.get(word, unk_id) for word in words]
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        # YOUR CODE HERE
        words = [self.id_to_word.get(id, self.unk_token) for id in ids]
        return " ".join(words)
        
