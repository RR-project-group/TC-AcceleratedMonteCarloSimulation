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
    model_params = {k: v for k, v in params.items() if k != 'true_price'}
    
    true_tensor = params.get('true_price')
    if true_tensor is not None:
        true_tensor = torch.tensor(true_tensor, dtype=model_params['dtype'], device=model_params.get('device', 'cpu'))

    for i in range(n_runs):
        try:
            pred = model_fn(**model_params)  # Call model to get predictions
            if pred is None:
                print(f"[Warning] Run {i+1}: Prediction is None — skipping.")
                continue

            err = compute_errors(true_tensor, pred)  # Compute error metrics
            records.append({'run_id': i, **err})  # Store results

        except Exception as e:
            print(f"[Error] Run {i+1} failed: {e}")
            continue

    return pd.DataFrame(records)
