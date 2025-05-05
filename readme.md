# Monte Carlo Simulation with PyTorch (Black-Scholes & LMM)

This project demonstrates how to accelerate financial Monte Carlo simulations using PyTorch. We implement both baseline and low-precision (bfloat16) variants to mimic Tensor Core-style acceleration.

## Features
- Black-Scholes option pricing with Monte Carlo simulation
- Libor Market Model (LMM) path simulation with Cholesky factorization
- Support for float32 and bfloat16 precision
- Performance benchmarking and precision comparison
- Visualization of error curves and execution times

## File Structure

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
