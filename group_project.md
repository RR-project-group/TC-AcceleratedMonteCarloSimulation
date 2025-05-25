# Group Project: Reproducing "Tensor Processing Units for Financial Monte Carlo"

## 📘 Project Summary

This is a team project for reproducing key ideas and experiments from the paper:

**"Tensor Processing Units for Financial Monte Carlo"**  
Andreas Rasch and Ludwig Gauckler, SIAM 2020  
[DOI: 10.1137/1.9781611976137.2](https://epubs.siam.org/doi/abs/10.1137/1.9781611976137.2)

We aim to recreate the numerical experiments in Tensorflow, focusing on:

- Monte Carlo simulation for financial models: **Black-Scholes (BS)** and **Libor Market Model (LMM)**
- Reproducing performance and accuracy benchmarks
- Simulating **Tensor Core (TC)** acceleration via bfloat16 computations
- Visualizing results (error curves, speed-up charts)

---

## 🎯 Project Goals

- ✅ Reproduce BS and LMM path simulations
- ✅ Implement baseline (float32) and TC-style (bfloat16) versions
- ✅ Compare accuracy between float32 and bfloat16
- ✅ Benchmark performance (speedup with low-precision)
- ✅ Produce plots for error and performance comparison

---

## 👥 Team Roles & Responsibilities

### 🧠 Team Lead

- Designs overall architecture and TC acceleration strategy
- Assists with financial model understanding (BS, LMM)
- Reviews and merges code via GitHub
- Coordinates interaction between implementation and testing members
- Oversees accuracy/performance consistency and documentation

---

### 👨‍🔬 Member A — **Accuracy Testing + Plotting**

- Implements error metrics: MSE, max error, relative error
- Compares float32 vs bfloat16 across BS & LMM
- Generates visualizations:
  - Accuracy curve (e.g., relative error vs. #simulations)
  - Precision histogram or heatmap
- Wraps evaluation code in reusable functions

---

### 🚀 Member B — **Performance Testing + Benchmarking**

- Builds timing functions (e.g., using `time`)
- Benchmarks:
  - Runtime under different simulation sizes
  - Float32 vs bfloat16 performance
- Outputs performance plots:
  - Time vs number of paths
  - Speedup ratio between precisions
- Optional: explore Tensorflow benchmarking tools

---

### 📈 Member C — **BS Simulation + Low-Precision Variant**

- Implements Monte Carlo path generation for Black-Scholes
- Adds support for float32 and bfloat16 precision
- Handles vectorization and batch simulation
- Collaborates with lead for TC-style logic (e.g., replacing matmuls with bfloat16 logic)

---

### 📉 Member D — **LMM Simulation + Cholesky + Low-Precision Variant**

- Implements LMM simulation: forward rate evolution, volatility modeling
- Includes Cholesky factorization for correlation matrices
- Supports float32 and bfloat16 precision variants
- Ensures compatibility with accuracy/performance scripts

---

## 🛠 Collaboration Guidelines

- Git strategy:
- Use feature branches: `feature/bs`, `test/accuracy`, etc.
- Weekly merge to `main` by team lead
- Dependencies:
- Python 3.9+, Tensorflow, Matplotlib
- Each member should:
- Write modular, testable code
- Document function inputs/outputs
- Provide minimal examples for their module

---

## 📎 Reference (APA Format)

Rasch, A., & Gauckler, L. (2020). *Tensor Processing Units for Financial Monte Carlo*. In Proceedings of the Platform for Advanced Scientific Computing Conference. [https://doi.org/10.1137/1.9781611976137.2](https://doi.org/10.1137/1.9781611976137.2)

---

Happy simulating! 💻📉📊