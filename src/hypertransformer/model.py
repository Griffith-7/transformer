"""Unified TransformerLanguageModel wrapping all three attention variants."""

import torch
import torch.nn as nn
from torch.nn import functional as F

from .models.adaptive import TransformerBlock as AdaptiveBlock
from .models.spiking_lorentz import TransformerBlock as SpikingLorentzBlock
from .models.standard import TransformerBlock as StandardBlock

BLOCK_MAP = {
    "standard": StandardBlock,
    "spiking_lorentz": SpikingLorentzBlock,
    "adaptive": AdaptiveBlock,
}


class TransformerLanguageModel(nn.Module):
    """Configurable language model supporting Standard, Spiking Lorentz,
    and Adaptive Hyperbolic attention mechanisms.

    Args:
        vocab_size: Size of the vocabulary.
        embed_dim: Embedding dimension.
        num_heads: Number of attention heads.
        num_layers: Number of transformer blocks.
        seq_len: Maximum sequence length.
        dropout: Dropout rate.
        variant: One of ``"standard"``, ``"spiking_lorentz"``, ``"adaptive"``.
    """

    def __init__(
        self,
        vocab_size,
        embed_dim=256,
        num_heads=4,
        num_layers=4,
        seq_len=128,
        dropout=0.1,
        variant="standard",
    ):
        super().__init__()
        self.config = {
            "vocab_size": vocab_size,
            "embed_dim": embed_dim,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "seq_len": seq_len,
            "dropout": dropout,
            "variant": variant,
        }

        block_cls = BLOCK_MAP[variant]
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(seq_len, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [block_cls(embed_dim, num_heads, dropout) for _ in range(num_layers)]
        )
        self.ln_f = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.token_embedding.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, mask=None):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(pos)
        x = self.dropout(tok_emb + pos_emb)
        for block in self.blocks:
            x = block(x, mask=mask)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.config["vocab_size"]), targets.reshape(-1)
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config["seq_len"] :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
