import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Define the directory where CSV files are stored
# Dynamically get the directory where the current script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Join the base directory with the 'result' folder to form an absolute path
DATA_DIR = os.path.join(BASE_DIR, 'LMMresult')  # Automatically adapts the path, no manual changes needed


# Map filenames to labels for plotting
file_map = {
    'lmm_fp32_cpu.csv': 'CPU-fp32',
    'lmm_fp32_cuda.csv': 'GPU-fp32',
    'lmm_fp32_tpu.csv': 'TPU-fp32', 
}

# Load and label data
all_data = []
for filename, label in file_map.items():
    path = os.path.join(DATA_DIR, filename)
    if os.path.exists(path):
        df = pd.read_csv(path)
        df['Setting'] = label
        all_data.append(df)
    else:
        print(f"Warning: {filename} not found.")

data = pd.concat(all_data, ignore_index=True)

# Plot 1: Boxplot for each error metric
metrics = ['MSE', 'Max_Abs_Error', 'Mean_Rel_Error']
for metric in metrics:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Setting', y=metric, data=data)
    plt.xticks(rotation=45)
    plt.title(f'LMM - {metric} Comparison Across Settings')
    plt.tight_layout()
    plt.savefig(os.path.join(DATA_DIR, f'LMM_{metric}_boxplot.png'))
    plt.close()

# Plot 2: Bar plot of average metrics
mean_data = data.groupby('Setting')[metrics].mean().reset_index()

plt.figure(figsize=(10, 6))
mean_data.plot(x='Setting', kind='bar', figsize=(12, 6))
plt.title('LMM - Average Error Metrics by Setting')
plt.ylabel('Error Value')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(DATA_DIR, 'LMM_average_error_barplot.png'))
plt.close()

# Plot 3: Line plot for MSE evolution 
plt.figure(figsize=(10, 6))
sns.lineplot(data=data, x='run_id', y='MSE', hue='Setting')
plt.title('LMM - MSE Over Runs')
plt.tight_layout()
plt.savefig(os.path.join(DATA_DIR, 'LMM_mse_over_runs.png'))
plt.close()

print("Plots saved: *_boxplot.png, average_error_barplot.png, mse_over_runs.png")