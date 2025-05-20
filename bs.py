
import torch
import torch.nn.functional as F

def black_scholes_mc(S0, K, T, r, sigma, N, dtype=torch.float32, device='cuda'):
    S0 = torch.tensor(S0, dtype=dtype, device=device)
    Z = torch.randn(N, dtype=dtype, device=device)
    ST = S0 * torch.exp((r - 0.5 * sigma ** 2) * T + sigma * torch.sqrt(torch.tensor(T, dtype=dtype)) * Z)
    payoff = F.relu(ST - K)
    return torch.exp(-r * T) * payoff.mean()

def run_bs_experiment(Ns, precision, device='cuda'):
    S0, K, T, r, sigma = 100, 100, 1, 0.05, 0.2
    dtype = torch.float32 if precision == 'fp32' else torch.bfloat16
    prices, times = [], []

    for N in Ns:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        price = black_scholes_mc(S0, K, T, r, sigma, N, dtype=dtype, device=device)
        end.record()
        torch.cuda.synchronize()

        elapsed_time = start.elapsed_time(end)
        prices.append(price.item())
        times.append(elapsed_time)

    return prices, times
