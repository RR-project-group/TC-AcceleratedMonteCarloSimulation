from accuracy_metrics import benchmark           # Import benchmarking utility to measure performance/accuracy
from bs import black_scholes_mc # Import the Black-Scholes Monte Carlo simulation implementation
import torch
import os  # Add this at the top if not already imported

# Automatically select device: use GPU if available, otherwise fallback to CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define parameters for the Black-Scholes model
params = {
    'S0': 100.0,             # Initial stock price
    'K': 105.0,              # Strike price
    'r': 0.05,               # Risk-free interest rate
    'sigma': 0.2,            # Volatility of the underlying asset
    'T': 1.0,                # Time to maturity in years
    'N': 10**6,        # Number of Monte Carlo simulation paths
    'true_price': 8.021352   # Reference price from Black-Scholes analytical formula
}

# Run benchmark using 32-bit floating point precision (float32)
df_fp32 = benchmark(
    black_scholes_mc,                          # Function to evaluate
    {**params, 'dtype': torch.float32, 'device': device},      # Parameters with dtype set to float32
    n_runs=50                                # Run the test 50 times for statistical stability
)

# Run benchmark using bfloat16 precision (lower precision)
df_bf16 = benchmark(
    black_scholes_mc,
    {**params, 'dtype': torch.bfloat16, 'device': device},     # Parameters with dtype set to bfloat16
    n_runs=50
)

# Define full path for the result folder
result_dir = "D:/GitProjects/RRProject/TC-AcceleratedMonteCarloSimulation/result"
# Ensure results directory exists
os.makedirs(result_dir, exist_ok=True)

# Save results into the specified result directory
df_fp32.to_csv(os.path.join(result_dir, 'bs_fp32.csv'), index=False)
df_bf16.to_csv(os.path.join(result_dir, 'bs_bf16.csv'), index=False)