from accuracy_metrics import compute_errors
import torch

def test_compute_errors():
    true = torch.tensor([1.0, 2.0, 3.0])         # Ground truth values
    pred = torch.tensor([1.1, 1.9, 3.05])        # Predicted values
    result = compute_errors(true, pred)         # Call the error computation function
    assert abs(result['MSE'] - 0.0025) < 1e-6    # Check if the MSE output is correct