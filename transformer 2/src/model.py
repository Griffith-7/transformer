import math
import torch
import torch.nn as nn
from torch.nn import functional as F

# ==========================================
# PHASE 3: SPIKING LORENTZ TRANSFORMER (SLT) - STANDARD VERSION
# ==========================================
# Testing: Replace Lorentz with Standard Attention
# to see if hyperbolic geometry actually helps
# ==========================================

class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, surprise_scores, threshold):
        scale = 4.0
        ctx.save_for_backward(surprise_scores, threshold)
        ctx.scale = scale
        return (surprise_scores > threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        surprise_scores, threshold = ctx.saved_tensors
        scale = ctx.scale
        sigmoid = torch.sigmoid((surprise_scores - threshold) * scale)
        grad_x = grad_output * sigmoid * (1 - sigmoid) * scale
        grad_threshold = (grad_output * (sigmoid - 0.5)).sum()
        return grad_x, grad_threshold


class SpikingStandardMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"
        
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.surprise_net = nn.Linear(embed_dim, 1)
        self.spike_threshold = nn.Parameter(torch.tensor(0.5))
        
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        B, T, C = x.size()
        
        # Surprise Logic (Optional - can be disabled)
        surprise_scores = torch.sigmoid(self.surprise_net(x)) 
        causal_mask = torch.tril(torch.ones_like(surprise_scores, dtype=torch.bool), diagonal=-1)
        surprise_scores_masked = surprise_scores.masked_fill(~causal_mask, 0.0)
        spikes = SurrogateSpike.apply(surprise_scores_masked, self.spike_threshold) 
        spike_mask = spikes.view(B, 1, T, 1) 
        
        # QKV Projections
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] 
        
        # STANDARD Euclidean Attention (like T1!)
        scores = torch.matmul(q, k.transpose(-2, -1))
        
        scale = math.sqrt(self.head_dim)
        scores = scores / scale
        
        if mask is None:
            mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device)).view(1, 1, T, T)
            
        scores = scores.masked_fill(~mask, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply the binary Spikes (optional - can be disabled)
        attn_weights = attn_weights * spike_mask
        
        attn_weights = self.attn_dropout(attn_weights)
        y = torch.matmul(attn_weights, v)
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)
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
        self.attn = SpikingStandardMultiHeadAttention(embed_dim, num_heads, dropout)
        self.ln_2 = nn.LayerNorm(embed_dim)
        self.ffwd = FeedForward(embed_dim, dropout)

    def forward(self, x, mask=None):
        x = x + self.attn(self.ln_1(x), mask=mask)
        x = x + self.ffwd(self.ln_2(x))
        return x

class TransformerLanguageModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, num_heads=4, num_layers=4, seq_len=128, dropout=0.1):
        super().__init__()
        
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(seq_len, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, dropout) for _ in range(num_layers)
        ])
        
        self.ln_f = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size, bias=False)
        
        # Weight tying - proven to help!
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
        x = tok_emb + pos_emb
        
        x = self.dropout(x)
        
        if mask is None:
            mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=idx.device)).view(1, 1, T, T)
            
        for block in self.blocks:
            x = block(x, mask=mask)
            
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            logits_flat = logits.view(B * T, -1)
            targets_flat = targets.view(B * T)
            loss = F.cross_entropy(logits_flat, targets_flat)
            
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx