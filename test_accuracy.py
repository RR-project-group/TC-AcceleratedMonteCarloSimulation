from accuracy_metrics import compute_errors
import torch

def test_compute_errors():
    # Define ground truth and predicted values
    true = torch.tensor([1.0, 2.0, 3.0])
    pred = torch.tensor([1.1, 1.9, 3.05])

    # Run the function to be tested
    result = compute_errors(true, pred)

    # Compute expected values dynamically
    abs_diff = torch.abs(pred - true)
    expected = {
        'MSE': torch.mean(abs_diff ** 2).item(),
        'Max_Abs_Error': torch.max(abs_diff).item(),
        'Mean_Rel_Error': torch.mean(abs_diff / torch.abs(true)).item()
    }

    # Define acceptable error thresholds for floating-point comparisons
    thresholds = {
        'MSE': 1e-6,
        'Max_Abs_Error': 1e-6,
        'Mean_Rel_Error': 1e-6
    }

    # Print and verify test results for each metric
    print("Running test for compute_errors():")
    for key in expected:
        diff = abs(result[key] - expected[key])
        passed = diff < thresholds[key]
        status = "PASS" if passed else "FAIL"
        print(f"  {key:15s}: result = {result[key]:.6f}, expected = {expected[key]:.6f}, {status}")

    # Ensure all checks passed; if not, raise an error
    assert all(abs(result[key] - expected[key]) < thresholds[key] for key in expected), "Test failed."

if __name__ == "__main__":
    test_compute_errors()
