# Evolution of Spiking Hyperbolic Transformers

This repository documents a three-phase research evolution from standard Euclidean transformers to advanced Hybrid-Manifold architectures.

## ⚡ Quick Start
1. **Install Requirements:** `pip install -r requirements.txt`
2. **Download Data:** `python scripts/download_data.py`
3. **Train T3:** `cd "transformer 3" && python train.py`

## 🔬 Scientific Overview
This project explores the intersection of **Riemannian Geometry** and **Neuromorphic Spiking Networks**. While standard Transformers (T1) remain the performance baseline on modern GPUs, our Adaptive Hyperbolic Turbo (T3) architecture investigates more efficient ways to encode linguistic hierarchies.

## 🚀 The Three Generations
| Model | Code Name | Primary Geometry | Key Feature |
| :--- | :--- | :--- | :--- |
| **Phase 1** | Standard | Euclidean | Industrial Baseline (Fastest) |
| **Phase 2** | SLT | Lorentz | Pure Hyperbolic reasoning (Minkowski optimized) |
| **Phase 3** | **AHT** | **Hybrid** | **Adaptive Minkowski + Spiking** |

## 📊 Honest Performance Trade-offs
- **Speed:** **Transformer 1** is the fastest on GPU due to fused CUDA kernels. **Transformer 2** is now highly optimized for CPU research (4.4x faster than previous versions).
- **Intelligence:** **Transformer 3** utilizes learnable curvature (`k`) and a learned blending gate (`alpha`) to navigate between flat and hierarchical data.
- **Stability:** All hyperbolic models include an **FP32 Safety Vault** to prevent NaNs in half-precision training.

## 🛠 Features (Turbo v5)
- **Minkowski Speedup:** Optimized O(T²) hyperbolic distance calculation.
- **Learnable Curvature (k):** Each attention head learns its own manifold curvature (softplus-stabilized).
- **Adaptive Blending:** A learned gate (`alpha`) that chooses the optimal geometry for every token.
- **Smart Checkpoints:** Model dimensions and vocabulary are auto-saved and auto-loaded by `generate.py`.

## 🏁 Conclusion
This repository is a research exploration. If you need raw throughput, use **Transformer 1**. If you are researching Geometric Deep Learning, **Transformer 3** represents the state-of-the-art for this framework.
