from accuracy_metrics import benchmark           # Import benchmarking utility to measure performance/accuracy
from bs import black_scholes_mc # Import the Black-Scholes Monte Carlo simulation implementation
from test_env import detect_environment # Import environment detection function (TPU, GPU/CUDA, CPU)
import torch # Import PyTorch for tensor operations
import os  # Import os module for file system operations

def run_bs_benchmark():
    """
    Run the Black-Scholes benchmark on the detected environment.
    Automatically selects device (TPU, GPU, or CPU), runs the benchmark
    for both float32 and bfloat16 precision, and saves results to CSV files.
    """
    env = detect_environment()  # Determine environment type

    # Setup torch device accordingly
    if env == 'tpu':
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
    else:
        device = torch.device(env)

    # Define model parameters
    params = {
        'S0': 100.0,            # Initial stock price
        'K': 120.0,             # Strike price
        'r': 0.05,              # Risk-free interest rate
        'sigma': 0.2,           # Volatility of the underlying asset
        'T': 1.0,               # Time to maturity in years
        'N': 10**6,             # Number of Monte Carlo simulation paths
        'true_price': 3.24747741656  # Reference price from analytical Black-Scholes formula
    }

    # Benchmark with float32 precision
    df_fp32 = benchmark(
        black_scholes_mc,
        {**params, 'dtype': torch.float32, 'device': device},
        n_runs=50
    )

    # Benchmark with bfloat16 precision
    df_bf16 = benchmark(
        black_scholes_mc,
        {**params, 'dtype': torch.bfloat16, 'device': device},
        n_runs=50
    )

    # Dynamically get path of current script and build result directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    result_dir = os.path.join(base_dir, 'BSresult')
    os.makedirs(result_dir, exist_ok=True)

    # Save results to CSV files, include device type in file name
    df_fp32.to_csv(os.path.join(result_dir, f"bs_fp32_{env}.csv"), index=False)
    df_bf16.to_csv(os.path.join(result_dir, f"bs_bf16_{env}.csv"), index=False)

    print(f"Results saved to: {result_dir}/bs_fp32_{env}.csv and bs_bf16_{env}.csv")


# Entry point when script is run directly
if __name__ == "__main__":
    run_bs_benchmark()