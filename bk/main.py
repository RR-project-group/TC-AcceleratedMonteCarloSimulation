from bs import run_bs_experiment
from lmm import run_lmm_experiment
from plot import plot_precision_curve, plot_performance

if __name__ == '__main__':
    import torch

    Ns = [10**i for i in range(3, 8)]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("Running Black-Scholes Benchmark...")
    prices_fp32, times_fp32 = run_bs_experiment(Ns, 'fp32', device=device)
    prices_bf16, times_bf16 = run_bs_experiment(Ns, 'bf16', device=device)

    plot_precision_curve(Ns, prices_fp32, prices_bf16)
    plot_performance(Ns, times_fp32, times_bf16)

    print("Running LMM Benchmark...")
    _, lmm_time_fp32 = run_lmm_experiment(1000000, 64, 10, 'fp32', device)
    _, lmm_time_bf16 = run_lmm_experiment(1000000, 64, 10, 'bf16', device)

    print(f"LMM Time fp32: {lmm_time_fp32:.2f}ms | bf16: {lmm_time_bf16:.2f}ms")
