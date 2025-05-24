import torch
import time

# Attempt to import PyTorch/XLA for TPU support
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    XLA_AVAILABLE = True
except ImportError:
    XLA_AVAILABLE = False


def generate_cov_matrix(num_factors, dim, dtype, device, eps=1e-1):
    """
    Generate a positive definite covariance matrix using:
    cov = A @ A.T + eps * I
    The 'eps' term ensures numerical stability, especially for low precision types.
    """
    A = torch.randn(dim, num_factors, dtype=dtype, device=device)
    cov = A @ A.T
    cov += eps * torch.eye(dim, dtype=dtype, device=device)
    return cov


def simulate_paths(L, n_paths, dtype, device):
    """
    Generate sample paths using the Cholesky factor L.
    Each path is sampled as: Z @ L.T, where Z ~ N(0, I)
    """
    Z = torch.randn(n_paths, L.shape[0], dtype=dtype, device=device)
    return Z @ L.T


def is_bfloat16_cholesky_supported(device):
    """
    Check if Cholesky decomposition is supported for bfloat16 on the given device.
    This is needed because many devices (especially GPUs) may not support it yet.
    """
    try:
        A = torch.randn(4, 4, dtype=torch.bfloat16, device=device)
        A = A @ A.T + torch.eye(4, dtype=torch.bfloat16, device=device) * 1e-1
        _ = torch.linalg.cholesky(A)
        return True
    except Exception:
        return False


def measure_time(func, device, *args, **kwargs):
    """
    Time a function call on different device types:
    - CUDA: uses torch.cuda.Event for precise measurement.
    - XLA: uses wall time with xm.mark_step() to flush computation.
    - CPU: uses regular time.time().
    """
    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = func(*args, **kwargs)
        end.record()
        torch.cuda.synchronize()
        elapsed = start.elapsed_time(end)
    elif device.type == "xla":
        t0 = time.time()
        result = func(*args, **kwargs)
        xm.mark_step()  # ensures computation is executed on TPU
        elapsed = (time.time() - t0) * 1000
    else:
        t0 = time.time()
        result = func(*args, **kwargs)
        elapsed = (time.time() - t0) * 1000
    return result, elapsed


def run_lmm_experiment(n_paths, dim, num_factors, precision, device='cuda'):
    """
    Run a Linear Market Model (LMM) simulation using Cholesky decomposition.
    
    Args:
        n_paths (int): Number of simulated paths.
        dim (int): Dimension of each path.
        num_factors (int): Number of latent factors in the covariance structure.
        precision (str): Either 'fp32' or 'bfloat16'.
        device (str): One of 'cuda', 'cpu', or 'xla' (TPU).
    
    Returns:
        paths (Tensor or None): Simulated paths (or None on failure).
        elapsed_time (float or None): Time in milliseconds (or None on failure).
    """
    # Select device: CPU, CUDA, or XLA (TPU)
    if device == 'xla':
        if not XLA_AVAILABLE:
            print("[Error] PyTorch/XLA not available. Install torch_xla.")
            return None, None
        device = xm.xla_device()
    else:
        device = torch.device(device)

    # Choose data type based on precision
    dtype = torch.float32 if precision == 'fp32' else torch.bfloat16

    # Check bfloat16 Cholesky support
    if dtype == torch.bfloat16 and not is_bfloat16_cholesky_supported(device):
        print(f"[Warning] bfloat16 run failed on {device}: Cholesky not supported")
        return None, None

    # Generate covariance matrix and perform Cholesky decomposition
    cov = generate_cov_matrix(num_factors, dim, dtype, device)
    try:
        L = torch.linalg.cholesky(cov)
    except RuntimeError as e:
        print(f"[Warning] Cholesky decomposition failed: {e}")
        return None, None

    # Simulate paths and measure elapsed time
    paths, elapsed_time = measure_time(simulate_paths, device, L, n_paths, dtype, device)
    return paths, elapsed_time