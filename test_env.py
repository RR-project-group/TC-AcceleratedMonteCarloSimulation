# Verify the environment
""" import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected") """
# Verify the environment
import torch

def detect_environment():
    print(f"PyTorch version: {torch.__version__}")
    
    # Try TPU first
    try:
        import torch_xla.core.xla_model as xm
        tpu_device = xm.xla_device()
        print("Environment: TPU")
        return
    except ImportError:
        print("TPU support not installed (torch_xla)")
    except RuntimeError:
        print("TPU not available")

    # Then check GPU
    if torch.cuda.is_available():
        print("Environment: GPU")
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
    else:
        print("Environment: CPU")

detect_environment()

