def compute_relative_error(ref, test):
    return abs((test - ref) / ref)

import matplotlib.pyplot as plt

def plot_precision_curve(Ns, prices_fp32, prices_bf16):
    errors = [abs(a - b) / a for a, b in zip(prices_fp32, prices_bf16)]
    plt.figure()
    plt.plot(Ns, errors, marker='o', label='Relative Error (bfloat16 vs fp32)')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Number of Paths')
    plt.ylabel('Relative Error')
    plt.title('Precision Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig('precision_curve.png')


def plot_performance(Ns, times_fp32, times_bf16):
    plt.figure()
    plt.plot(Ns, times_fp32, marker='o', label='fp32')
    plt.plot(Ns, times_bf16, marker='x', label='bfloat16')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Number of Paths')
    plt.ylabel('Time (ms)')
    plt.title('Performance Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig('performance_comparison.png')
