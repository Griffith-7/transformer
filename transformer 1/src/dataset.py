import os
import torch
from torch.utils.data import Dataset
from collections import Counter
import pickle

class Tokenizer:
    def __init__(self, max_vocab_size=20000):
        self.max_vocab_size = max_vocab_size
        self.stoi = {}
        self.itos = {}
        self.pad_token = '<pad>'
        self.unk_token = '<unk>'
        self.eos_token = '<eos>'
        
    def build_vocab(self, file_path):
        print(f"Building vocabulary from {file_path}...")
        counter = Counter()
        
        # Count words iteratively to save memory
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # wikitext tokens are space separated space
                words = line.strip().split()
                if not words:
                    continue
                counter.update(words)
                counter.update([self.eos_token])
                
        # Keep most common words up to max_vocab_size - 3 (for special tokens)
        common_words = counter.most_common(self.max_vocab_size - 3)
        
        # Initialize vocab with special tokens
        self.itos = {
            0: self.pad_token,
            1: self.unk_token,
            2: self.eos_token
        }
        self.stoi = {v: k for k, v in self.itos.items()}
        
        # Add common words to vocab
        for word, _ in common_words:
            idx = len(self.stoi)
            self.stoi[word] = idx
            self.itos[idx] = word
            
        print(f"Vocabulary built with {len(self.stoi)} tokens.")
        
    def encode(self, text):
        return [self.stoi.get(w, self.stoi[self.unk_token]) for w in text.split()]
        
    def decode(self, indices):
        return " ".join([self.itos.get(idx, self.unk_token) for idx in indices])
        
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({'stoi': self.stoi, 'itos': self.itos}, f)
            
    def load(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.stoi = data['stoi']
            self.itos = data['itos']

class WikiTextDataset(Dataset):
    def __init__(self, file_path, tokenizer, seq_len=128):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.tokens = []
        
        print(f"Tokenizing {file_path}...")
        # Since full wikitext-103 is large, we load it into an integer list
        # For ~100M tokens, Python list might take some memory, but it's manageable (few GB max).
        # We can optimize by using an array.array or torch tensor if needed.
        
        import array
        # Use unsigned short if vocab < 65535, otherwise unsigned int
        typecode = 'H' if len(tokenizer.stoi) <= 65535 else 'I'
        self.tokens = array.array(typecode)
        
        eos_id = tokenizer.stoi[tokenizer.eos_token]
        unk_id = tokenizer.stoi[tokenizer.unk_token]
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                words = line.strip().split()
                if not words:
                    continue
                # encode
                encoded = [tokenizer.stoi.get(w, unk_id) for w in words]
                encoded.append(eos_id)
                self.tokens.extend(encoded)
                
        # Convert to torch tensor for faster slicing in __getitem__
        # Using long tensor type which is int64 as expected by nn.Embedding
        self.tokens = torch.tensor(self.tokens, dtype=torch.long)
        print(f"Dataset created with {len(self.tokens)} total tokens.")

    def __len__(self):
        # We need sequence + 1 to get targets covering the shift
        # Divide by seq_len to jump in non-overlapping chunks!
        return (len(self.tokens) - self.seq_len - 1) // self.seq_len

    def __getitem__(self, idx):
        # inputs: [start_idx : start_idx + seq_len]
        # targets: [start_idx + 1 : start_idx + seq_len + 1]
        start_idx = idx * self.seq_len
        x = self.tokens[start_idx : start_idx + self.seq_len]
        y = self.tokens[start_idx + 1 : start_idx + self.seq_len + 1]
        return x, y
