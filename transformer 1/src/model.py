import math
import torch
import torch.nn as nn
from torch.nn import functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        B, T, C = x.size()
        
        # Calculate query, key, value for all heads in batch and move head forward to be the batch dim
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] # (B, num_heads, T, head_dim)
        
        # Causal Attention using PyTorch scaled_dot_product_attention if available (FlashAttention)
        is_causal = mask is None # if mask is None, we assume causal masking for LM
        
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.attn_dropout.p if self.training else 0, is_causal=is_causal)
        
        # Re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        
        # output projection
        y = self.resid_dropout(self.out_proj(y))
        return y


class FeedForward(nn.Module):
    def __init__(self, embed_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.ln_2 = nn.LayerNorm(embed_dim)
        self.ffwd = FeedForward(embed_dim, dropout)

    def forward(self, x, mask=None):
        x = x + self.attn(self.ln_1(x), mask=mask)
        x = x + self.ffwd(self.ln_2(x))
        return x


class TransformerLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, num_heads=4, num_layers=4, seq_len=256, dropout=0.1):
        super().__init__()
        
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        
        # Vocab embedding
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        # Positional embedding
        self.position_embedding = nn.Embedding(seq_len, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout) for _ in range(num_layers)
        ])
        
        self.ln_f = nn.LayerNorm(embed_dim)
        
        # LM Head
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        
        # Weight tying (optional but good practice)
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
        assert T <= self.seq_len, f"Cannot forward sequence of length {T}, block size is only {self.seq_len}"
        
        # Token and Positional embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        tok_emb = self.token_embedding(idx) # (B, T, C)
        pos_emb = self.position_embedding(pos) # (T, C)
        
        x = self.dropout(tok_emb + pos_emb)
        
        # Pass through Transformer blocks
        for block in self.blocks:
            x = block(x, mask=mask)
            
        x = self.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        
        loss = None
        if targets is not None:
            # Flatten to calculate cross-entropy simpler
            B, T, C = logits.size()
            logits_flat = logits.reshape(B * T, C)
            targets_flat = targets.reshape(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)
            
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # Crop context if it gets too long
            idx_cond = idx[:, -self.seq_len:]
            
            # Forward pass
            logits, _ = self(idx_cond)
            
            # Focus only on the last time step
            logits = logits[:, -1, :] / temperature
            
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)
            
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)
            
        return idx
