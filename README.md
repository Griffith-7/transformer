# Evolution of Spiking Hyperbolic Transformers

This repository documents a three-phase research evolution from standard Euclidean transformers to advanced Hybrid-Manifold architectures.

## 🔬 Scientific Overview
This project explores the intersection of **Riemannian Geometry** and **Neuromorphic Spiking Networks**. While standard Transformers (T1) remain the performance baseline on modern GPUs, our Adaptive Hyperbolic Turbo (T3) architecture investigates more efficient ways to encode linguistic hierarchies.

## 🚀 The Three Generations
| Model | Code Name | Primary Geometry | Key Feature |
| :--- | :--- | :--- | :--- |
| **Phase 1** | Standard | Euclidean | Industrial Baseline (Fastest) |
| **Phase 2** | SLT | Lorentz | Pure Hyperbolic reasoning |
| **Phase 3** | **AHT** | **Hybrid** | **Adaptive Minkowski + Spiking** |

## 📊 Honest Performance Trade-offs
A third-party audit comparing these architectures on WikiText-2 reveals the following:

- **Speed:** **Transformer 1** is significantly faster (3x-5x) because it uses fused CUDA kernels (`scaled_dot_product_attention`). Transformer 3 uses custom manifold math which is more computationally intensive.
- **Complexity:** **Transformer 3** is the most sophisticated, utilizing learnable per-head curvature (`k`) and a learned blending gate (`alpha`) to navigate between flat and hierarchical data.
- **Efficiency:** All models in this repository now implement **Weight Tying** and **AMP (FP16)** to maximize parameter utility and hardware speed.

## 🛠 Features (Turbo v5)
- **Minkowski Speedup:** Optimized O(T²) hyperbolic distance calculation (recovering speed from previous custom implementations).
- **Learnable Curvature (k):** Each attention head learns its own manifold curvature (softplus-stabilized).
- **Adaptive Blending:** A learned gate (`alpha`) that chooses the optimal geometry for every token.
- **Hardened Stability:** Implements L2-normalization pre-projection and tight gradient clipping to prevent manifold instability.

## 🏁 Conclusion
This repository is a research exploration. If you need raw throughput, use **Transformer 1**. If you are researching Geometric Deep Learning or Energy-Efficient Spiking Transformers, **Transformer 3** represents the state-of-the-art for this framework.
