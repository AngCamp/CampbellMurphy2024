import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from fitter import Fitter
import json
import pandas as pd

# Paths
DATA_DIR = "/space/scratch/SWR_final_pipeline/validation_data_figure9"
OUT_DIR = os.path.dirname(__file__)

DATASETS = [
    ("ABI Behaviour", "abi_visbehave_swr_theta_speed.npz"),
    ("ABI Coding", "abi_viscoding_swr_theta_speed.npz"),
    ("IBL", "ibl_swr_theta_speed.npz"),
]

# Plot settings
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["text.usetex"] = False
plt.rcParams["font.weight"] = "bold"
plt.rcParams["font.size"] = 8
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"

# Create figure
fig, axes = plt.subplots(4, 3, figsize=(12, 16))
fig.suptitle("Figure 9: SWR Event Properties", fontsize=12, fontweight="bold")

ks_results = {}

for col, (title, fname) in enumerate(DATASETS):
    data_path = os.path.join(DATA_DIR, fname)
    if not os.path.exists(data_path):
        print(f"Missing data: {data_path}")
        continue
    
    # Load data
    arr = np.load(data_path, allow_pickle=True)["results"]
    df = pd.DataFrame(list(arr))
    dataset_key = fname.split('_swr')[0]
    ks_results[dataset_key] = {}
    
    # Row 1: Histogram of mean theta power (z-score)
    ax = axes[0, col]
    ax.hist(df["mean_theta_power"].dropna(), bins=20, edgecolor="black", facecolor="steelblue", alpha=0.7)
    ax.set_xlabel("Mean Theta Power (z-score)", fontweight="bold")
    ax.set_ylabel("Count", fontweight="bold")
    ax.set_title(f"{title}", fontweight="bold")
    ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(4))
    
    # Row 2: Histogram of mean speed
    ax = axes[1, col]
    ax.hist(df["mean_speed"].dropna(), bins=20, edgecolor="black", facecolor="lightcoral", alpha=0.7)
    ax.set_xlabel("Mean Speed (cm/s)", fontweight="bold")
    ax.set_ylabel("Count", fontweight="bold")
    ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(4))
    
    # Row 3: Density plot of duration
    ax = axes[2, col]
    durations = df["duration"].dropna() * 1000  # Convert to ms
    f_duration = Fitter(durations, distributions=['norm', 'halfnorm', 'lognorm'])
    f_duration.fit()
    best_dist_duration = f_duration.get_best(method='sumsquare_error')
    f_duration.hist()
    f_duration.plot_pdf()
    ax.set_xlabel("Duration (ms)", fontweight="bold")
    ax.set_ylabel("Density", fontweight="bold")
    ax.grid(False)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(4))
    
    # Store KS results for duration
    if 'lognorm' in f_duration.fitted_param:
        ks_results[dataset_key]['duration'] = {
            'sse': f_duration.summary()['sumsquare_error']['lognorm'],
            'ks_stat': f_duration.summary()['ks_stat']['lognorm'],
            'ks_pvalue': f_duration.summary()['ks_pvalue']['lognorm']
        }
    
    # Row 4: Density plot of peak ripple power (if available)
    ax = axes[3, col]
    if 'power_max_zscore' in df.columns:
        peak_power = df["power_max_zscore"].dropna()
        f_power = Fitter(peak_power, distributions=['norm', 'halfnorm', 'lognorm'])
        f_power.fit()
        best_dist_power = f_power.get_best(method='sumsquare_error')
        f_power.hist()
        f_power.plot_pdf()
        ax.set_xlabel("Peak Ripple Power (z-score)", fontweight="bold")
        ax.set_ylabel("Density", fontweight="bold")
        ax.grid(False)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
        ax.xaxis.set_major_locator(ticker.MaxNLocator(4))
        
        # Store KS results for peak power
        if 'lognorm' in f_power.fitted_param:
            ks_results[dataset_key]['peak_power'] = {
                'sse': f_power.summary()['sumsquare_error']['lognorm'],
                'ks_stat': f_power.summary()['ks_stat']['lognorm'],
                'ks_pvalue': f_power.summary()['ks_pvalue']['lognorm']
            }
    else:
        ax.text(0.5, 0.5, "Peak power data\nnot available", ha='center', va='center', transform=ax.transAxes, fontweight="bold")
        ax.set_xlabel("Peak Ripple Power (z-score)", fontweight="bold")
        ax.set_ylabel("Density", fontweight="bold")

plt.tight_layout()

# Save figure
plt.savefig(os.path.join(OUT_DIR, "figure9.png"), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(OUT_DIR, "figure9.svg"), bbox_inches='tight')

# Save KS results
with open(os.path.join(OUT_DIR, "figure9_ks_results.json"), 'w') as f:
    json.dump(ks_results, f, indent=2)

print(f"Figure saved to {OUT_DIR}")
print(f"KS results saved to {os.path.join(OUT_DIR, 'figure9_ks_results.json')}") 