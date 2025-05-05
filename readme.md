# Monte Carlo Simulation with PyTorch (Black-Scholes & LMM)

This project demonstrates how to accelerate financial Monte Carlo simulations using PyTorch. We implement both baseline and low-precision (bfloat16) version. We also leaverage Tensor Core acceleration for the execution speed optimization.

## Features
- Black-Scholes option pricing with Monte Carlo simulation
- Libor Market Model (LMM) path simulation with Cholesky factorization
- Support for float32 and bfloat16 precision
- Performance benchmarking and precision comparison
- Visualization of error curves and execution times

## Usage
1. Install dependencies:
```bash
pip install torch matplotlib
```

2. Run the benchmark:
```bash
python main.py
```

3. Outputs:
- `precision_curve.png`: Visualizes relative error between bfloat16 and float32
- `performance_comparison.png`: Runtime comparison of both precisions

## Notes
- Ensure a CUDA-capable GPU or bf16 precision supported TPU is available for best results
- If running on CPU-only, modify the `device` to `'cpu'` in `main.py`

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