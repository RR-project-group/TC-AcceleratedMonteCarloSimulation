import torch
import pandas as pd

def compute_errors(true: torch.Tensor, pred: torch.Tensor) -> dict:
    """
    Compute core error metrics between predicted and true values.

    Parameters:
    - true (torch.Tensor): Ground truth values.
    - pred (torch.Tensor): Predicted values (same shape as `true`).

    Returns:
    - dict: Dictionary containing:
        - 'MSE': Mean Squared Error
        - 'Max_Abs_Error': Maximum Absolute Error
        - 'Mean_Rel_Error': Mean Relative Error
    """
    abs_diff = torch.abs(pred - true)
    return {
        'MSE': torch.mean(abs_diff ** 2).item(),
        'Max_Abs_Error': torch.max(abs_diff).item(),
        'Mean_Rel_Error': torch.mean(abs_diff / torch.abs(true)).item()
    }

def benchmark(model_fn, params: dict, n_runs: int = 10) -> pd.DataFrame:
    """
    Run multiple evaluations of a model function and collect error metrics.

    Parameters:
    - model_fn (callable): A function that returns predicted prices as a torch.Tensor.
    - params (dict): Parameters to pass to the model function. Must include key 'true_price'.
    - n_runs (int): Number of runs to repeat the evaluation (default = 10).

    Returns:
    - pd.DataFrame: A DataFrame with each run's ID and corresponding error metrics.
    """
    records = []
    for i in range(n_runs):
        pred = model_fn(**params)  # Call model to get predictions
        err = compute_errors(params['true_price'], pred)  # Compute error metrics
        records.append({'run_id': i, **err})  # Store results
    return pd.DataFrame(records)
