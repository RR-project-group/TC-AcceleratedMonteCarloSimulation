# Monte Carlo Simulation with Tensorflow (Black-Scholes & LMM)

This project demonstrates how to accelerate financial Monte Carlo simulations using Tensorflow. We implement both baseline and low-precision (bfloat16) version. We also leaverage Tensor Core acceleration for the execution speed optimization.

## Features
- Black-Scholes option pricing with Monte Carlo simulation
- Libor Market Model (LMM) path simulation with Cholesky factorization
- Support for float32 and bfloat16 precision
- Performance benchmarking and precision comparison
- Visualization of error curves and execution times

## Usage
1. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RR-project-group/TC-AcceleratedMonteCarloSimulation/blob/main/main.ipynb)

2. To connect to a TPU runtime in Google Colab, follow these steps:
- Go to the menu bar, click on Runtime → Change runtime type.
- In the popup window, under Hardware accelerator, select TPU.
- Click Save.
- After that, your notebook is connected to a TPU runtime. You can verify by running:

3. Outputs:
- Visualizes relative error between bfloat16 and float32
- Runtime comparison of both precisions

## Notes
- Ensure a Colab TPU backend is available for best results

## Reference

This project is a reproduction of this work:

**Rasch, A., & Gauckler, L. (2020). Tensor Processing Units for Financial Monte Carlo.**  
*In International Conference on High Performance Computing (pp. 1–12).*  
[SIAM DOI: 10.1137/1.9781611976137.2](https://epubs.siam.org/doi/abs/10.1137/1.9781611976137.2)
<details>
<summary>BibTeX</summary>
```bibtex
@inproceedings{rasch2020tpu,
  author = {Rasch, Andreas and Gauckler, Ludwig},
  title = {Tensor Processing Units for Financial Monte Carlo},
  booktitle = {Proceedings of the Platform for Advanced Scientific Computing Conference},
  year = {2020},
  publisher = {SIAM},
  doi = {10.1137/1.9781611976137.2},
  url = {https://epubs.siam.org/doi/abs/10.1137/1.9781611976137.2}
}