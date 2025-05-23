import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    import numpy as np
except ImportError:
    install("numpy")
    import numpy as np

try:
    from scipy.stats import norm
except ImportError:
    install("scipy")
    from scipy.stats import norm

# Black-Scholes analytic pricing function
def bs_analytic(S0, K, r, sigma, T):
    """
    Black-Scholes closed-form solution for a European call option.

    Parameters:
    S0    -- initial stock price
    K     -- strike price
    r     -- risk-free interest rate (annualized)
    sigma -- volatility of the underlying asset (annualized)
    T     -- time to maturity (in years)

    Returns:
    Theoretical price of the European call option.
    """
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

# Optional test: Vanilla european call option (reference example)
if __name__ == "__main__":
    S0, K, r, sigma, T = 100, 120, 0.05, 0.2, 1.0
    true_price = bs_analytic(S0, K, r, sigma, T)
    print(f"Black-Scholes Analytical Price: {true_price:.11f}") # 3.24747741656(TPU&GPU&CPU) 