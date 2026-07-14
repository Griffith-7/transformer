# Architecture

## Overview

HyperTransformer implements three successive generations of transformer attention mechanisms, each building on the last to explore the intersection of **Riemannian Geometry** and **Neuromorphic Spiking Networks**.

```
┌─────────────────────────────────────────────────────────────────┐
│                    TransformerLanguageModel                      │
│                                                                 │
│  ┌─────────┐    ┌──────────┐    ┌──────────────┐               │
│  │ Token   │ +  │ Position │ →  │   Dropout    │               │
│  │ Embed   │    │ Embed    │    └──────┬───────┘               │
│  └─────────┘    └──────────┘           │                        │
│                                        ▼                        │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │              TransformerBlock  (× num_layers)             │  │
│  │                                                          │  │
│  │  LayerNorm → [Attention Variant] → Residual              │  │
│  │  LayerNorm → FeedForward(GELU)    → Residual              │  │
│  │                                                          │  │
│  │  ┌────────────────────────────────────────────────────┐  │  │
│  │  │          Attention Variant Selection                │  │  │
│  │  │                                                    │  │  │
│  │  │  T1: MultiHeadAttention        (Euclidean SDPA)   │  │  │
│  │  │  T2: SpikingLorentzAttention   (Minkowski dist)   │  │  │
│  │  │  T3: AdaptiveGeometryAttention (Hybrid blend)     │  │  │
│  │  └────────────────────────────────────────────────────┘  │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                        │                        │
│                                        ▼                        │
│                              LayerNorm → LM Head               │
└─────────────────────────────────────────────────────────────────┘
```

## Three Variants

### T1 — Standard Transformer
- **Geometry:** Euclidean (flat)
- **Attention:** `F.scaled_dot_product_attention` (PyTorch fused CUDA kernels)
- **Key trait:** Fastest on GPU, industrial baseline

### T2 — Spiking Lorentz Transformer
- **Geometry:** Lorentz (hyperbolic, constant curvature k=1)
- **Attention:** Minkowski inner product → geodesic distance via `torch.acosh`
- **Key trait:** Surrogate gradient spiking for energy-efficient attention routing
- **Safety:** FP32 vault for all hyperbolic operations (prevents NaN in FP16)

### T3 — Adaptive Hyperbolic Transformer (AHT)
- **Geometry:** Learnable hybrid (Euclidean + Hyperbolic per-head)
- **Attention:** Adaptive blending gate `α` mixes Euclidean and Lorentz scores
- **Key traits:**
  - Per-head learnable curvature `k` via `softplus(log_k)`
  - QK normalization scale via learned parameter
  - Surrogate gradient spiking (same as T2)

## Surrogate Gradient Spiking

All hyperbolic variants (T2, T3) use a surrogate gradient for the non-differentiable spike operation:

```
Forward:  spike(x) = 1 if x > threshold, else 0
Backward: ∂L/∂x = sigmoid_grad(scaled difference) * scale
```

This enables end-to-end training through the binary spike decision.
