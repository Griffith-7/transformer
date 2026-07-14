"""Tests for dataset.py and Tokenizer across all three transformer versions."""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "transformer 1"))
from src.dataset import Tokenizer, WikiTextDataset


@pytest.fixture
def tiny_corpus(tmp_path):
    """Create a tiny text file for testing."""
    data_path = tmp_path / "train.txt"
    lines = []
    for _ in range(200):
        lines.append("the cat sat on the mat and the dog ran in the park")
        lines.append("a bird flew over the tree and sang a song")
    data_path.write_text("\n".join(lines))
    return str(data_path)


@pytest.fixture
def tiny_valid(tmp_path):
    data_path = tmp_path / "valid.txt"
    lines = ["the cat sat on the mat"] * 50
    data_path.write_text("\n".join(lines))
    return str(data_path)


class TestTokenizer:
    def test_build_vocab_no_eos_collision(self, tiny_corpus):
        """Bug 1 regression: <eos> must not be re-added at a higher index."""
        tok = Tokenizer(max_vocab_size=100)
        tok.build_vocab(tiny_corpus)

        eos_idx = tok.stoi["<eos>"]
        assert eos_idx == 2, f"<eos> should be at index 2, got {eos_idx}"
        assert len(tok.stoi) == len(tok.itos), "stoi and itos must have same length"
        assert max(tok.itos.keys()) + 1 == len(tok.itos), "itos indices must be contiguous"

    def test_vocab_size_bounds(self, tiny_corpus):
        """All token IDs must be < vocab_size."""
        tok = Tokenizer(max_vocab_size=100)
        tok.build_vocab(tiny_corpus)
        vocab_size = len(tok.stoi)
        for token, idx in tok.stoi.items():
            assert 0 <= idx < vocab_size, (
                f"Token '{token}' has index {idx} >= vocab_size {vocab_size}"
            )

    def test_encode_decode_roundtrip(self, tiny_corpus):
        tok = Tokenizer(max_vocab_size=100)
        tok.build_vocab(tiny_corpus)
        text = "the cat sat"
        encoded = tok.encode(text)
        decoded = tok.decode(encoded)
        assert decoded == text

    def test_encode_unknown_returns_unk(self, tiny_corpus):
        tok = Tokenizer(max_vocab_size=100)
        tok.build_vocab(tiny_corpus)
        encoded = tok.encode("xyzzy foobar")
        assert all(idx == tok.stoi["<unk>"] for idx in encoded)

    def test_save_load_roundtrip(self, tiny_corpus, tmp_path):
        tok = Tokenizer(max_vocab_size=100)
        tok.build_vocab(tiny_corpus)
        save_path = str(tmp_path / "vocab.pkl")
        tok.save(save_path)

        tok2 = Tokenizer(max_vocab_size=100)
        tok2.load(save_path)
        assert tok.stoi == tok2.stoi
        assert tok.itos == tok2.itos

    def test_max_vocab_size_respected(self, tiny_corpus):
        tok = Tokenizer(max_vocab_size=10)
        tok.build_vocab(tiny_corpus)
        assert len(tok.stoi) <= 10


class TestWikiTextDataset:
    def test_dataset_length(self, tiny_corpus):
        tok = Tokenizer(max_vocab_size=100)
        tok.build_vocab(tiny_corpus)
        ds = WikiTextDataset(tiny_corpus, tok, seq_len=16)
        assert len(ds) > 0

    def test_sample_shapes(self, tiny_corpus):
        tok = Tokenizer(max_vocab_size=100)
        tok.build_vocab(tiny_corpus)
        ds = WikiTextDataset(tiny_corpus, tok, seq_len=16)
        x, y = ds[0]
        assert x.shape == (16,)
        assert y.shape == (16,)

    def test_target_is_next_token(self, tiny_corpus):
        tok = Tokenizer(max_vocab_size=100)
        tok.build_vocab(tiny_corpus)
        ds = WikiTextDataset(tiny_corpus, tok, seq_len=16)
        x, y = ds[0]
        assert torch.all(y[:-1] == x[1:]), "Target should be shifted by 1"

    def test_no_out_of_range_tokens(self, tiny_corpus):
        """All dataset token IDs must be valid vocab indices."""
        tok = Tokenizer(max_vocab_size=100)
        tok.build_vocab(tiny_corpus)
        ds = WikiTextDataset(tiny_corpus, tok, seq_len=16)
        vocab_size = len(tok.stoi)
        for i in range(min(len(ds), 50)):
            x, y = ds[i]
            assert x.max() < vocab_size, f"x has out-of-range token at sample {i}"
            assert y.max() < vocab_size, f"y has out-of-range token at sample {i}"

    def test_dataloader_batch(self, tiny_corpus):
        from torch.utils.data import DataLoader

        tok = Tokenizer(max_vocab_size=100)
        tok.build_vocab(tiny_corpus)
        ds = WikiTextDataset(tiny_corpus, tok, seq_len=16)
        loader = DataLoader(ds, batch_size=8, shuffle=True)
        x, y = next(iter(loader))
        assert x.shape == (8, 16)
        assert y.shape == (8, 16)
