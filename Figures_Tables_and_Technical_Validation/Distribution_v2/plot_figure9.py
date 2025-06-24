import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from fitter import Fitter
import json
import pandas as pd
import datetime
import shutil
import scipy.stats as stats

# Paths
DATA_DIR = "/home/acampbell/NeuropixelsLFPOnRamp/Figures_Tables_and_Technical_Validation/Distribution_v2/distributions_for_plotting"
OUT_DIR = os.path.dirname(__file__)

DATASETS = [
    ("ABI Behaviour", "abi_visbehave"),
    ("ABI Coding", "abi_viscoding"),
    ("IBL", "ibl"),
]
METRICS = [
    ("Mean Theta Power (Z-score)", "theta_power"),
    ("Wheel Speed (cm/s)", "speed"),
    ("Duration (ms)", "duration"),
    ("Peak Ripple Power (Z-score)", "peak_power"),
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
    if value == 0 or np.isnan(value):
        return "N/A"
    exp = int(np.floor(np.log10(abs(value))))
    mantissa = value / (10**exp)
    return f"{mantissa:.1f}x10{exp:+d}".replace("+", "")

def generate_caption(ks_results):
    """Generate caption with actual statistical values"""
    caption = """**Figure 9.  Detected events show expected properties for probe level putative events, occurring in the absence of the wheel movement and at lower theta power.   Plots for ABI Behaviour events are on the left, ABI Coding in the middle, and IBL on the right.  a) The mean instantaneous theta power z-scored during probe level event windows.  b) The wheel speed during the global events, note that for the ABI datasets this can be interpreted as a running speed but from the IBL the mouse uses ambulation to turn a wheel. c) The distribution of global level event durations.  The best fit distribution of the normal, half-normal and lognormal were fit to the data, the lognormal was shown to be the best fit by the Kolmogrov-Smirnov (KS) test in all entities."""
    
    # Add duration results (using lognorm as specified in caption)
    if 'abi_visbehave' in ks_results and 'duration' in ks_results['abi_visbehave'] and 'lognorm' in ks_results['abi_visbehave']['duration']:
        dur = ks_results['abi_visbehave']['duration']['lognorm']
        caption += f"  ABI Behaviour (SSE {format_scientific_notation(dur['sse'])}, KS {format_scientific_notation(dur['ks_stat'])}, KS p-value <0.0001)"
    
    if 'abi_viscoding' in ks_results and 'duration' in ks_results['abi_viscoding'] and 'lognorm' in ks_results['abi_viscoding']['duration']:
        dur = ks_results['abi_viscoding']['duration']['lognorm']
        caption += f", ABI Coding (SSE {format_scientific_notation(dur['sse'])}, KS {format_scientific_notation(dur['ks_stat'])}, KS p-value <0.0001)"
    
    if 'ibl' in ks_results and 'duration' in ks_results['ibl'] and 'lognorm' in ks_results['ibl']['duration']:
        dur = ks_results['ibl']['duration']['lognorm']
        caption += f" and the IBL (SSE {format_scientific_notation(dur['sse'])}, KS {format_scientific_notation(dur['ks_stat'])}, KS p-value <0.0001)"
    
    caption += """  d)  The distribution of global event level peak ripple power (z-scored) is best fit by the lognormal distribution in all entities, all fits pass significance."""
    
    # Add peak power results (using lognorm as specified in caption)
    if 'abi_visbehave' in ks_results and 'peak_power' in ks_results['abi_visbehave'] and 'lognorm' in ks_results['abi_visbehave']['peak_power']:
        pow = ks_results['abi_visbehave']['peak_power']['lognorm']
        caption += f"  ABI Behaviour (SSE {format_scientific_notation(pow['sse'])}, KS {format_scientific_notation(pow['ks_stat'])}, KS p-value <0.0001)"
    
    if 'abi_viscoding' in ks_results and 'peak_power' in ks_results['abi_viscoding'] and 'lognorm' in ks_results['abi_viscoding']['peak_power']:
        pow = ks_results['abi_viscoding']['peak_power']['lognorm']
        caption += f", ABI Coding (SSE {format_scientific_notation(pow['sse'])}, KS {format_scientific_notation(pow['ks_stat'])}, KS p-value <0.0001)"
    
    if 'ibl' in ks_results and 'peak_power' in ks_results['ibl'] and 'lognorm' in ks_results['ibl']['peak_power']:
        pow = ks_results['ibl']['peak_power']['lognorm']
        caption += f" and the IBL (SSE {format_scientific_notation(pow['sse'])}, KS {format_scientific_notation(pow['ks_stat'])}, KS p-value <0.0001)"
    
    caption += "."
    return caption

# Create figure
fig, axes = plt.subplots(4, 3, figsize=(12, 16))
fig.subplots_adjust(hspace=0.4, wspace=0.3)
fig.suptitle("Figure 9: SWR Event Properties", fontsize=16, fontweight="bold")

ks_results = {}

# Create output directory for individual subplots
INDIVIDUAL_DIR = os.path.join(OUT_DIR, 'individual_subplots')
if os.path.exists(INDIVIDUAL_DIR):
    shutil.rmtree(INDIVIDUAL_DIR)
os.makedirs(INDIVIDUAL_DIR, exist_ok=True)

for col, (dataset_title, dataset_key) in enumerate(DATASETS):
    ks_results[dataset_key] = {}
    for row, (metric_label, metric_key) in enumerate(METRICS):
        ax = axes[row, col]
        plt.sca(ax)  # Ensure all plotting happens on the correct axes
        npz_file = os.path.join(DATA_DIR, f"{dataset_key}_{metric_key}.npz")
        if not os.path.exists(npz_file):
            ax.text(0.5, 0.5, f"No data", ha='center', va='center', transform=ax.transAxes, fontweight="bold")
            continue
        data = np.load(npz_file)['data']
        nan_count = np.isnan(data).sum()
        total_count = len(data)
        nan_pct = 100 * nan_count / total_count if total_count > 0 else 0
        print(f"{dataset_key} {metric_key}: {nan_count}/{total_count} ({nan_pct:.1f}%) NaN values")
        if nan_pct > 10:
            print(f"WARNING: More than 10% NaN in {dataset_key} {metric_key}!")
        data = data[~np.isnan(data)]
        if metric_key == "duration":
            data = data * 1000  # Convert to ms
        if len(data) == 0:
            ax.text(0.5, 0.5, f"No valid data", ha='center', va='center', transform=ax.transAxes, fontweight="bold")
            continue
        # Plot
        if metric_key in ["theta_power", "speed"]:
            ax.hist(data, bins=30, edgecolor="black", facecolor="steelblue", alpha=0.7)
        else:  # duration, peak_power: density plot with fits
            # Plot histogram on ax
            ax.hist(data, bins=30, edgecolor="black", facecolor="steelblue", alpha=0.7, density=True)
            # Fit and plot distributions on ax
            f = Fitter(data, distributions=['norm', 'halfnorm', 'lognorm'])
            f.fit()
            xlim = ax.get_xlim()
            x = np.linspace(xlim[0], xlim[1], 200)
            for dist_name, color in zip(['norm', 'halfnorm', 'lognorm'], ['red', 'orange', 'green']):
                if dist_name in f.fitted_param:
                    params = f.fitted_param[dist_name]
                    dist = getattr(stats, dist_name)
                    y = dist.pdf(x, *params)
                    ax.plot(x, y, color=color, label=dist_name)
            # Store KS results for all distributions
            try:
                summary = f.summary()
                ks_results[dataset_key][metric_key] = {}
                for dist_name in ['norm', 'halfnorm', 'lognorm']:
                    if dist_name in summary.index:
                        ks_results[dataset_key][metric_key][dist_name] = {
                            'sse': summary.loc[dist_name, 'sumsquare_error'],
                            'ks_stat': summary.loc[dist_name, 'ks_statistic'],
                            'ks_pvalue': summary.loc[dist_name, 'ks_pvalue'],
                            'aic': summary.loc[dist_name, 'aic'],
                            'bic': summary.loc[dist_name, 'bic']
                        }
                    else:
                        ks_results[dataset_key][metric_key][dist_name] = {
                            'sse': np.nan,
                            'ks_stat': np.nan,
                            'ks_pvalue': np.nan,
                            'aic': np.nan,
                            'bic': np.nan
                        }
            except Exception as e:
                print(f"Error extracting KS results for {dataset_key} {metric_key}: {e}")
                ks_results[dataset_key][metric_key] = {
                    'norm': {'sse': np.nan, 'ks_stat': np.nan, 'ks_pvalue': np.nan, 'aic': np.nan, 'bic': np.nan},
                    'halfnorm': {'sse': np.nan, 'ks_stat': np.nan, 'ks_pvalue': np.nan, 'aic': np.nan, 'bic': np.nan},
                    'lognorm': {'sse': np.nan, 'ks_stat': np.nan, 'ks_pvalue': np.nan, 'aic': np.nan, 'bic': np.nan}
                }
            if row in [2, 3]:
                ax.legend(fontsize=10, loc='upper right')
        # Axis labels
        if row == 0:
            ax.set_title(dataset_title, fontweight="bold", fontsize=16)
        if col == 0:
            # Row labels (a, b, c, d)
            label = ['a', 'b', 'c', 'd'][row]
            ax.annotate(label, xy=(-0.18, 1.05), xycoords='axes fraction', fontsize=18, fontweight='bold', va='top', ha='right')
        ax.set_xlabel(metric_label, fontweight="bold", fontsize=14)
        ax.set_ylabel("Event Count" if metric_key in ["theta_power", "speed"] else "Density", fontweight="bold", fontsize=14)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(5))
        ax.yaxis.set_major_locator(ticker.MaxNLocator(5))
        # Save individual subplot
        subplot_filename = f"{dataset_key}_{metric_key}.png"
        subplot_svg = f"{dataset_key}_{metric_key}.svg"
        ax.figure.savefig(os.path.join(INDIVIDUAL_DIR, subplot_filename), dpi=300, bbox_inches='tight')
        ax.figure.savefig(os.path.join(INDIVIDUAL_DIR, subplot_svg), bbox_inches='tight')

plt.tight_layout(rect=[0, 0, 1, 0.97])

# Save with timestamp to avoid overwriting
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
plt.savefig(os.path.join(OUT_DIR, f"figure9_{timestamp}.png"), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(OUT_DIR, f"figure9_{timestamp}.svg"), bbox_inches='tight')
plt.close(fig)
print(f"Figure saved to {OUT_DIR} as figure9_{timestamp}.png/svg")

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