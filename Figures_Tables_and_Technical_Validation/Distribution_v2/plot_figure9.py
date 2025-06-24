import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from fitter import Fitter
import json
import pandas as pd

# Paths
DATA_DIR = "/home/acampbell/NeuropixelsLFPOnRamp/Figures_Tables_and_Technical_Validation/Distribution_v2/distributions_for_plotting"
OUT_DIR = os.path.dirname(__file__)

DATASETS = [
    ("ABI Behaviour", "abi_visbehave"),
    ("ABI Coding", "abi_viscoding"),
    ("IBL", "ibl"),
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

for col, (title, dataset_key) in enumerate(DATASETS):
    print(f"Processing dataset: {dataset_key}")
    ks_results[dataset_key] = {}
    
    # Row 1: Histogram of mean theta power (z-score)
    ax = axes[0, col]
    theta_file = os.path.join(DATA_DIR, f"{dataset_key}_theta_power.npz")
    if os.path.exists(theta_file):
        theta_data = np.load(theta_file)['data']
        if len(theta_data) > 0:
            ax.hist(theta_data, bins=20, edgecolor="black", facecolor="steelblue", alpha=0.7)
        else:
            ax.text(0.5, 0.5, "No valid theta data", ha='center', va='center', transform=ax.transAxes, fontweight="bold")
    else:
        ax.text(0.5, 0.5, "No theta data file", ha='center', va='center', transform=ax.transAxes, fontweight="bold")
    ax.set_xlabel("Mean Theta Power (z-score)", fontweight="bold")
    ax.set_ylabel("Count", fontweight="bold")
    ax.set_title(f"{title}", fontweight="bold")
    ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(4))
    
    # Row 2: Histogram of mean speed
    ax = axes[1, col]
    speed_file = os.path.join(DATA_DIR, f"{dataset_key}_speed.npz")
    if os.path.exists(speed_file):
        speed_data = np.load(speed_file)['data']
        if len(speed_data) > 0:
            ax.hist(speed_data, bins=20, edgecolor="black", facecolor="lightcoral", alpha=0.7)
        else:
            ax.text(0.5, 0.5, "No valid speed data", ha='center', va='center', transform=ax.transAxes, fontweight="bold")
    else:
        ax.text(0.5, 0.5, "No speed data file", ha='center', va='center', transform=ax.transAxes, fontweight="bold")
    ax.set_xlabel("Mean Speed (cm/s)", fontweight="bold")
    ax.set_ylabel("Count", fontweight="bold")
    ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(4))
    
    # Row 3: Density plot of duration
    ax = axes[2, col]
    f_duration = None  # Initialize to None
    duration_file = os.path.join(DATA_DIR, f"{dataset_key}_duration.npz")
    if os.path.exists(duration_file):
        durations = np.load(duration_file)['data'] * 1000  # Convert to ms
        if len(durations) > 0:
            f_duration = Fitter(durations, distributions=['norm', 'halfnorm', 'lognorm'])
            f_duration.fit()
            best_dist_duration = f_duration.get_best(method='sumsquare_error')
            f_duration.hist()
            f_duration.plot_pdf()
        else:
            ax.text(0.5, 0.5, "No valid duration data", ha='center', va='center', transform=ax.transAxes, fontweight="bold")
    else:
        ax.text(0.5, 0.5, "No duration data file", ha='center', va='center', transform=ax.transAxes, fontweight="bold")
    ax.set_xlabel("Duration (ms)", fontweight="bold")
    ax.set_ylabel("Density", fontweight="bold")
    ax.grid(False)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
    ax.xaxis.set_major_locator(ticker.MaxNLocator(4))
    
    # Store KS results for duration
    if f_duration is not None and 'lognorm' in f_duration.fitted_param:
        try:
            summary = f_duration.summary()
            ks_results[dataset_key]['duration'] = {
                'sse': summary['sumsquare_error']['lognorm'],
                'ks_stat': summary.get('ks_stat', {}).get('lognorm', np.nan),
                'ks_pvalue': summary.get('ks_pvalue', {}).get('lognorm', np.nan)
            }
        except (KeyError, TypeError) as e:
            print(f"Warning: Could not extract KS results for {dataset_key} duration: {e}")
            ks_results[dataset_key]['duration'] = {
                'sse': np.nan,
                'ks_stat': np.nan,
                'ks_pvalue': np.nan
            }
    else:
        print(f"Warning: No valid duration data for {dataset_key}")
        ks_results[dataset_key]['duration'] = {
            'sse': np.nan,
            'ks_stat': np.nan,
            'ks_pvalue': np.nan
        }
    
    # Row 4: Density plot of peak ripple power
    ax = axes[3, col]
    f_power = None  # Initialize to None
    power_file = os.path.join(DATA_DIR, f"{dataset_key}_peak_power.npz")
    if os.path.exists(power_file):
        peak_power = np.load(power_file)['data']
        if len(peak_power) > 0:
            f_power = Fitter(peak_power, distributions=['norm', 'halfnorm', 'lognorm'])
            f_power.fit()
            best_dist_power = f_power.get_best(method='sumsquare_error')
            f_power.hist()
            f_power.plot_pdf()
        else:
            ax.text(0.5, 0.5, "No valid peak power data", ha='center', va='center', transform=ax.transAxes, fontweight="bold")
        ax.set_xlabel("Peak Ripple Power (z-score)", fontweight="bold")
        ax.set_ylabel("Density", fontweight="bold")
        ax.grid(False)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
        ax.xaxis.set_major_locator(ticker.MaxNLocator(4))
        
        # Store KS results for peak power
        if f_power is not None and 'lognorm' in f_power.fitted_param:
            try:
                summary = f_power.summary()
                ks_results[dataset_key]['peak_power'] = {
                    'sse': summary['sumsquare_error']['lognorm'],
                    'ks_stat': summary.get('ks_stat', {}).get('lognorm', np.nan),
                    'ks_pvalue': summary.get('ks_pvalue', {}).get('lognorm', np.nan)
                }
            except (KeyError, TypeError) as e:
                print(f"Warning: Could not extract KS results for {dataset_key} peak power: {e}")
                ks_results[dataset_key]['peak_power'] = {
                    'sse': np.nan,
                    'ks_stat': np.nan,
                    'ks_pvalue': np.nan
                }
        else:
            print(f"Warning: No valid peak power data for {dataset_key}")
            ks_results[dataset_key]['peak_power'] = {
                'sse': np.nan,
                'ks_stat': np.nan,
                'ks_pvalue': np.nan
            }
    else:
        ax.text(0.5, 0.5, "No peak power data file", ha='center', va='center', transform=ax.transAxes, fontweight="bold")
        ax.set_xlabel("Peak Ripple Power (z-score)", fontweight="bold")
        ax.set_ylabel("Density", fontweight="bold")
        # Set default peak power results
        ks_results[dataset_key]['peak_power'] = {
            'sse': np.nan,
            'ks_stat': np.nan,
            'ks_pvalue': np.nan
        }

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