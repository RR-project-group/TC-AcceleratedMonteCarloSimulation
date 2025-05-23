import torch
import pandas as pd
import os
from test_env import detect_environment # Import environment detection function (TPU, GPU/CUDA, CPU)


def generate_cov_matrix(num_factors, dim, device, eps=1e-3):
    """
    Generate a positive-definite covariance matrix using A @ A.T + eps * I.
    This ensures numerical stability even in low-precision (bfloat16).
    """
    A = torch.randn(dim, num_factors, dtype=torch.float32, device=device)
    cov = A @ A.T
    cov += eps * torch.eye(dim, dtype=torch.float32, device=device)
    return cov

def simulate_groundtruth_paths(n_paths=10000, dim=10, num_factors=3, device='cuda'):
    """
    Simulate LMM paths using float32 precision to generate ground truth.
    Returns: simulated paths (torch.Tensor)
    """
    cov = generate_cov_matrix(num_factors, dim, device)
    L = torch.linalg.cholesky(cov)

    Z = torch.randn(n_paths, dim, dtype=torch.float32, device=device)
    paths = Z @ L.T
    return paths

if __name__ == "__main__":

    env = detect_environment()  # Determine environment type

    groundtruth = simulate_groundtruth_paths(device=env)

    # Dynamically get path of current script and build result directory
    base_dir = os.path.dirname(os.path.abspath(__file__))
    result_dir = os.path.join(base_dir, 'LMMresult')
    os.makedirs(result_dir, exist_ok=True)

    # Save to file
    save_path = os.path.join(result_dir, f"lmm_groundtruth_{env}.csv")
    df = pd.DataFrame(groundtruth.cpu().numpy())
    df.to_csv(save_path, index=False)

    print(f"Saved ground truth LMM paths to {save_path}")
