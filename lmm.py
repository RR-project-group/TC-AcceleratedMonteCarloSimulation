import torch

def generate_cov_matrix(num_factors, dim, dtype, device):
    A = torch.randn(dim, num_factors, dtype=dtype, device=device)
    return A @ A.T

def simulate_paths(L, n_paths, dtype, device):
    Z = torch.randn(n_paths, L.shape[0], dtype=dtype, device=device)
    return Z @ L.T

def run_lmm_experiment(n_paths, dim, num_factors, precision, device='cuda'):
    dtype = torch.float32 if precision == 'fp32' else torch.bfloat16
    cov = generate_cov_matrix(num_factors, dim, dtype, device)
    L = torch.linalg.cholesky(cov)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    paths = simulate_paths(L, n_paths, dtype, device)
    end.record()
    torch.cuda.synchronize()

    elapsed_time = start.elapsed_time(end)
    return paths, elapsed_time
