# Verify the environment
import torch

def detect_environment() -> str:
    """Detects and returns the available hardware environment for PyTorch.
    
    Returns:
        str: The detected environment type - 'tpu', 'cuda' (GPU), or 'cpu'.
    """
    # Print PyTorch version for debugging/info purposes
    print(f"PyTorch version: {torch.__version__}")
    
    # Check for TPU (Tensor Processing Unit) availability first
    try:
        import torch_xla.core.xla_model as xm
        tpu_device = xm.xla_device()  # Attempt to get TPU device
        print("Environment: TPU")
        return 'tpu'  # Return early if TPU is found
    except ImportError:
        print("TPU support not installed (torch_xla)")
    except RuntimeError:
        print("TPU not available")

    # Check for CUDA (NVIDIA GPU) availability
    if torch.cuda.is_available():
        print("Environment: GPU")
        # Print the name of the first available GPU
        print(f"GPU name: {torch.cuda.get_device_name(0)}") 
        return 'cuda'  # Return if GPU is available
    
    # Default case - CPU fallback
    print("Environment: CPU")
    return 'cpu'

# detect_environment()

