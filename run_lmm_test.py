import torch
import os
import pandas as pd
from accuracy_metrics import benchmark
from test_env import detect_environment
from lmm import run_lmm_experiment

def run_lmm_benchmark():
    # Detect runtime environment: CPU, GPU, or TPU
    env = detect_environment()

    # Set torch device accordingly
    if env == 'tpu':
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
    else:
        device = torch.device(env)

    # Experiment configuration
    params = {
        'n_paths': 100000,
        'dim': 16,
        'num_factors': 8
    }

    # Load ground truth data (float32-based reference)
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        base_dir = os.getcwd()

    result_dir = os.path.join(base_dir, 'LMMresult')
    os.makedirs(result_dir, exist_ok=True)

    gt_path = os.path.join(result_dir, f"lmm_groundtruth_cpu.csv")

    groundtruth = torch.tensor(
        pd.read_csv(gt_path).values,
        dtype=torch.float32,
        device=device
    )

    # === Benchmark float32 ===
    paths_fp32 = None
    def wrapper_fp32(n_paths, dim, num_factors, dtype, device):
        nonlocal paths_fp32
        precision = 'fp32' if dtype == torch.float32 else 'bf16'
        paths_fp32, _ = run_lmm_experiment(n_paths, dim, num_factors, precision, device=device)
        return paths_fp32


    df_fp32 = benchmark(
        wrapper_fp32,
        {**params, 'dtype': torch.float32, 'device': device, 'true_price': groundtruth},
        n_runs=50
    )

    # === Benchmark bfloat16 ===
    paths_bf16 = None
    def wrapper_bf16(n_paths, dim, num_factors, dtype, device):
        nonlocal paths_bf16
        precision = 'fp32' if dtype == torch.float32 else 'bf16'
        try:
            paths_bf16, _ = run_lmm_experiment(n_paths, dim, num_factors, precision, device=device)
            return paths_bf16
        except RuntimeError as e:
            print(f"[Warning] bfloat16 run failed on {env}: {e}")
            return torch.zeros_like(groundtruth)


    df_bf16 = benchmark(
        wrapper_bf16,
        {**params, 'dtype': torch.bfloat16, 'device': device, 'true_price': groundtruth},
        n_runs=50
    )

    # === Save benchmark results and generated paths ===
    df_fp32.to_csv(os.path.join(result_dir, f"lmm_fp32_{env}.csv"), index=False)
    df_bf16.to_csv(os.path.join(result_dir, f"lmm_bf16_{env}.csv"), index=False)

#     if paths_fp32 is not None:
#         torch.save(paths_fp32.cpu(), os.path.join(result_dir, f"paths_fp32_{env}.pt"))
#     if paths_bf16 is not None:
#         torch.save(paths_bf16.cpu(), os.path.join(result_dir, f"paths_bf16_{env}.pt")) 

    print(f"Results saved to {result_dir}")

""" if __name__ == "__main__":
    run_lmm_benchmark() """
