"""Spiking Lorentz attention (Phase 2) — hyperbolic manifold + surrogate gradient spiking."""
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
import geoopt

from .spike import SurrogateSpike


class SpikingLorentzAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.surprise_net = nn.Linear(embed_dim, 1)
        self.spike_threshold = nn.Parameter(torch.tensor(0.5))
        self.manifold = geoopt.Lorentz(k=1.0)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, T, C = x.size()
        importance = torch.sigmoid(self.surprise_net(x))
        spikes = SurrogateSpike.apply(importance, self.spike_threshold)
        spike_mask = spikes.view(B, 1, 1, T)

        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        with torch.amp.autocast('cuda', enabled=False):
            q_32, k_32 = q.float(), k.float()
            q_hyp = self.manifold.expmap0(q_32)
            k_hyp = self.manifold.expmap0(k_32)
            q_time, q_space = q_hyp[..., 0:1], q_hyp[..., 1:]
            k_time, k_space = k_hyp[..., 0:1], k_hyp[..., 1:]
            inner_product = -torch.matmul(q_time, k_time.transpose(-2, -1)) + \
                             torch.matmul(q_space, k_space.transpose(-2, -1))
            minkowski_dot = torch.clamp(-inner_product, min=1.0 + 1e-6)
            scores = -(torch.acosh(minkowski_dot) ** 2) / math.sqrt(self.head_dim)
            scores = scores.to(q.dtype)

        if mask is None:
            mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device)).view(1, 1, T, T)
        self_attn = torch.eye(T, dtype=torch.bool, device=x.device).view(1, 1, T, T)
        combined_mask = (mask & (spike_mask > 0)) | self_attn
        scores = scores.masked_fill(~combined_mask, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)
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
        self.attn = SpikingLorentzAttention(embed_dim, num_heads, dropout)
        self.ln_2 = nn.LayerNorm(embed_dim)
        self.ffwd = FeedForward(embed_dim, dropout)

    def forward(self, x, mask=None):
        x = x + self.attn(self.ln_1(x), mask=mask)
        x = x + self.ffwd(self.ln_2(x))
        return x
