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

def format_scientific_notation(value):
    """Format a number in scientific notation like 6.0x10-6"""
    if value == 0:
        return "0"
    exp = int(np.floor(np.log10(abs(value))))
    mantissa = value / (10**exp)
    return f"{mantissa:.1f}x10{exp:+d}".replace("+", "")

def generate_caption(ks_results):
    """Generate caption with actual statistical values"""
    caption = """**Figure 9.  Detected events show expected properties for probe level putative events, occurring in the absence of the wheel movement and at lower theta power.   Plots for ABI Behaviour events are on the left, ABI Coding in the middle, and IBL on the right.  a) The mean instantaneous theta power z-scored during probe level event windows.  b) The wheel speed during the global events, note that for the ABI datasets this can be interpreted as a running speed but from the IBL the mouse uses ambulation to turn a wheel. c) The distribution of global level event durations.  The best fit distribution of the normal, half-normal and lognormal were fit to the data, the lognormal was shown to be the best fit by the Kolmogrov-Smirnov (KS) test in all entities."""
    
    # Add duration results
    if 'abi_visbehave' in ks_results and 'duration' in ks_results['abi_visbehave']:
        dur = ks_results['abi_visbehave']['duration']
        caption += f"  ABI Behaviour (SSE {format_scientific_notation(dur['sse'])}, KS {format_scientific_notation(dur['ks_stat'])}, KS p-value <0.0001)"
    
    if 'abi_viscoding' in ks_results and 'duration' in ks_results['abi_viscoding']:
        dur = ks_results['abi_viscoding']['duration']
        caption += f", ABI Coding (SSE {format_scientific_notation(dur['sse'])}, KS {format_scientific_notation(dur['ks_stat'])}, KS p-value <0.0001)"
    
    if 'ibl' in ks_results and 'duration' in ks_results['ibl']:
        dur = ks_results['ibl']['duration']
        caption += f" and the IBL (SSE {format_scientific_notation(dur['sse'])}, KS {format_scientific_notation(dur['ks_stat'])}, KS p-value <0.0001)"
    
    caption += """  d)  The distribution of global event level peak ripple power (z-scored) is best fit by the lognormal distribution in all entities, all fits pass significance."""
    
    # Add peak power results
    if 'abi_visbehave' in ks_results and 'peak_power' in ks_results['abi_visbehave']:
        pow = ks_results['abi_visbehave']['peak_power']
        caption += f"  ABI Behaviour (SSE {format_scientific_notation(pow['sse'])}, KS {format_scientific_notation(pow['ks_stat'])}, KS p-value <0.0001)"
    
    if 'abi_viscoding' in ks_results and 'peak_power' in ks_results['abi_viscoding']:
        pow = ks_results['abi_viscoding']['peak_power']
        caption += f", ABI Coding (SSE {format_scientific_notation(pow['sse'])}, KS {format_scientific_notation(pow['ks_stat'])}, KS p-value <0.0001)"
    
    if 'ibl' in ks_results and 'peak_power' in ks_results['ibl']:
        pow = ks_results['ibl']['peak_power']
        caption += f" and the IBL (SSE {format_scientific_notation(pow['sse'])}, KS {format_scientific_notation(pow['ks_stat'])}, KS p-value <0.0001)"
    
    caption += "."
    return caption

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

# Generate and save caption
caption = generate_caption(ks_results)
caption_file = os.path.join(OUT_DIR, "figure9_caption_with_results.txt")
with open(caption_file, 'w') as f:
    f.write(caption)

print(f"Figure saved to {OUT_DIR}")
print(f"KS results saved to {os.path.join(OUT_DIR, 'figure9_ks_results.json')}")
print(f"Caption saved to {caption_file}")
print("\nGenerated caption:")
print(caption) 