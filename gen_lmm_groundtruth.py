import torch
import pandas as pd
import os
from test_env import detect_environment
from lmm import generate_cov_matrix, simulate_paths

def simulate_groundtruth_paths(n_paths=100000, dim=16, num_factors=8, device='cuda'):
    cov = generate_cov_matrix(num_factors, dim, dtype=torch.float32, device=device)
    L = torch.linalg.cholesky(cov)
    return simulate_paths(L, n_paths, dtype=torch.float32, device=device)

def generate_and_save_groundtruth(n_paths=100000, dim=16, num_factors=8, save_dir='LMMresult', filename_prefix='lmm_groundtruth', device=None):
    """
    Generate LMM ground truth data and save to CSV file.

    Returns:
        save_path (str): Full path to saved CSV file.
        groundtruth (torch.Tensor): Simulated LMM paths.
    """
    env = device or detect_environment()
    groundtruth = simulate_groundtruth_paths(n_paths=n_paths, dim=dim, num_factors=num_factors, device=env)

    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError:
        base_dir = os.getcwd()

    result_dir = os.path.join(base_dir, 'LMMresult')
    os.makedirs(result_dir, exist_ok=True)

    save_path = os.path.join(result_dir, f"{filename_prefix}_{env}.csv")
    df = pd.DataFrame(groundtruth.cpu().numpy())
    df.to_csv(save_path, index=False)

    print(f"Saved ground truth LMM paths to {save_path}")
    return save_path, groundtruth
# save_path, gt_tensor = generate_and_save_groundtruth()