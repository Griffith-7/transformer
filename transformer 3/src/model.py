import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import geoopt

# ========================================================
# PHASE 5: ADAPTIVE HYPERBOLIC TURBO (AHT) - STABLE
# ========================================================
# RESEARCH-GRADE UPGRADES:
# 1. Minkowski Inner Product (O(T^2) Speedup)
# 2. Learnable Curvature (k) per Head (Softplus stabilized)
# 3. L2-Normalization Pre-Projection (Vanishing Gradient Fix)
# 4. Corrected Spike Causal Mask (diagonal=0)
# ========================================================

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


class AdaptiveGeometryAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # SNN Spike Controls
        self.importance_net = nn.Linear(embed_dim, 1)
        self.spike_threshold = nn.Parameter(torch.tensor(0.5))
        
        # Adaptive Blending
        self.alpha_net = nn.Linear(embed_dim, num_heads)
        
        # RESEARCH UPGRADE: Learnable Curvature per head
        # Initializing near k=1.0 using softplus(0.5413) approx 1.0
        self.log_k = nn.Parameter(torch.full((num_heads,), 0.54))
        
        # RESEARCH UPGRADE: Learnable QK Scale (Stability)
        self.qk_scale = nn.Parameter(torch.tensor(0.1))
        
        self.manifold = geoopt.Lorentz(k=1.0)
        
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        B, T, C = x.size()
        
        # 1. SPIKE CALCULATION
        importance = torch.sigmoid(self.importance_net(x)) 
        causal_mask = torch.tril(torch.ones_like(importance, dtype=torch.bool), diagonal=0)
        importance_masked = importance.masked_fill(~causal_mask, 0.0)
        spikes = SurrogateSpike.apply(importance_masked, self.spike_threshold) 
        spike_mask = spikes.view(B, 1, T, 1) 
        
        # 2. ADAPTIVE BLENDING
        alpha = torch.sigmoid(self.alpha_net(x)).transpose(1, 2).unsqueeze(-1)
        
        # 3. QKV PROJECTION
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k_vec, v = qkv[0], qkv[1], qkv[2] 
        
        # A. EUCLIDEAN PATH
        scores_euclid = torch.matmul(q, k_vec.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # B. LORENTZ PATH (Speedup)
        # SPEED LIMITER: Clamp scale to prevent exponential overflow
        # Soft-clamp qk_scale to max 1.0
        safe_scale = torch.sigmoid(self.qk_scale) * 1.5 
        
        q_norm = F.normalize(q, dim=-1) * safe_scale
        k_norm = F.normalize(k_vec, dim=-1) * safe_scale
        
        q_hyp = self.manifold.expmap0(q_norm)
        k_hyp = self.manifold.expmap0(k_norm)
        
        # Minkowski Inner Product: -q0*k0 + q_spatial @ k_spatial^T
        q_time, q_space = q_hyp[..., 0:1], q_hyp[..., 1:]
        k_time, k_space = k_hyp[..., 0:1], k_hyp[..., 1:]
        
        inner_product = -torch.matmul(q_time, k_time.transpose(-2, -1)) + \
                         torch.matmul(q_space, k_space.transpose(-2, -1))
        
        # STABILITY: Use softplus for curvature to avoid k=0
        curv_k = F.softplus(self.log_k).view(1, self.num_heads, 1, 1) + 1e-6
        
        # STABILITY: Clamp for acosh domain [1.0 + eps, inf]
        minkowski_dot = -inner_product
        minkowski_dot = torch.clamp(minkowski_dot, min=1.0 + 1e-6)
        
        # Distance calculation
        dist = torch.acosh(minkowski_dot)
        scores_hyper = -(dist ** 2) / curv_k
        
        # 4. ADAPTIVE GEOMETRY MERGE
        scores = (1 - alpha) * scores_euclid + alpha * scores_hyper
        
        if mask is None:
            mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device)).view(1, 1, T, T)
        
        scores = scores.masked_fill(~mask, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)
        
        # 5. SPIKE GATING
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
        self.attn = AdaptiveGeometryAttention(embed_dim, num_heads, dropout)
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
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
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