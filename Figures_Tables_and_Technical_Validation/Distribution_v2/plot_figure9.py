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
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

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

# Create output directory for individual subplots
INDIVIDUAL_DIR = os.path.join(OUT_DIR, 'individual_subplots')
if os.path.exists(INDIVIDUAL_DIR):
    shutil.rmtree(INDIVIDUAL_DIR)
os.makedirs(INDIVIDUAL_DIR, exist_ok=True)

def format_scientific_notation(value):
    """Format value in scientific notation for caption"""
    if np.isnan(value):
        return "NaN"
    if value == 0:
        return "0"
    
    exp = int(np.floor(np.log10(abs(value))))
    mantissa = value / (10**exp)
    
    if exp == 0:
        return f"{mantissa:.1f}"
    elif exp == -1:
        return f"{mantissa:.1f}x10-1"
    elif exp == -2:
        return f"{mantissa:.1f}x10-2"
    elif exp == -3:
        return f"{mantissa:.1f}x10-3"
    elif exp == -4:
        return f"{mantissa:.1f}x10-4"
    elif exp == -5:
        return f"{mantissa:.1f}x10-5"
    elif exp == -6:
        return f"{mantissa:.1f}x10-6"
    else:
        return f"{mantissa:.1f}x10{exp}"

def generate_caption(ks_results):
    """Generate caption with actual statistical values"""
    caption = """**Figure 9.  Detected events show expected properties for probe level putative events, occurring in the absence of the wheel movement and at lower theta power.   Plots for ABI Behaviour events are on the left, ABI Coding in the middle, and IBL on the right.  a) The mean instantaneous theta power z-scored during probe level event windows.  b) The wheel speed during the global events, note that for the ABI datasets this can be interpreted as a running speed but from the IBL the mouse uses ambulation to turn a wheel. c) The distribution of global level event durations.  The best fit distribution of the normal, half-normal and lognormal were fit to the data, the lognormal was shown to be the best fit by the Kolmogrov-Smirnov (KS) test in all entities.  ABI Behaviour (SSE {}, KS {}, KS p-value <0.0001), ABI Coding (SSE {}, KS {}, KS p-value <0.0001) and the IBL (SSE {}, KS {}, KS p-value <0.0001)  d)  The distribution of global event level peak ripple power (z-scored) is best fit by the lognormal distribution in all entities, all fits pass significance.  ABI Behaviour (SSE {}, KS {}, KS p-value <0.0001), ABI Coding (SSE {}, KS {}, KS p-value <0.0001) and the IBL (SSE {}, KS {}, KS p-value <0.0001).""".format(
        format_scientific_notation(ks_results['abi_visbehave']['duration']['lognorm']['sse']),
        format_scientific_notation(ks_results['abi_visbehave']['duration']['lognorm']['ks_stat']),
        format_scientific_notation(ks_results['abi_viscoding']['duration']['lognorm']['sse']),
        format_scientific_notation(ks_results['abi_viscoding']['duration']['lognorm']['ks_stat']),
        format_scientific_notation(ks_results['ibl']['duration']['lognorm']['sse']),
        format_scientific_notation(ks_results['ibl']['duration']['lognorm']['ks_stat']),
        format_scientific_notation(ks_results['abi_visbehave']['peak_power']['lognorm']['sse']),
        format_scientific_notation(ks_results['abi_visbehave']['peak_power']['lognorm']['ks_stat']),
        format_scientific_notation(ks_results['abi_viscoding']['peak_power']['lognorm']['sse']),
        format_scientific_notation(ks_results['abi_viscoding']['peak_power']['lognorm']['ks_stat']),
        format_scientific_notation(ks_results['ibl']['peak_power']['lognorm']['sse']),
        format_scientific_notation(ks_results['ibl']['peak_power']['lognorm']['ks_stat'])
    )
    return caption

# Initialize results dictionary
ks_results = {}

# Create individual figures for each subplot
individual_figures = []

for col, (dataset_title, dataset_key) in enumerate(DATASETS):
    ks_results[dataset_key] = {}
    for row, (metric_label, metric_key) in enumerate(METRICS):
        print(f"Processing {dataset_key} {metric_key}...")
        
        npz_file = os.path.join(DATA_DIR, f"{dataset_key}_{metric_key}.npz")
        if not os.path.exists(npz_file):
            print(f"Warning: {npz_file} not found")
            continue
            
        # Load data
        data = np.load(npz_file)['data']
        
        # Check for NaN values
        nan_count = np.isnan(data).sum()
        nan_percent = (nan_count / len(data)) * 100
        print(f"{dataset_key} {metric_key}: {nan_count}/{len(data)} ({nan_percent:.1f}%) NaN values")
        
        if nan_percent > 10:
            print(f"Warning: {nan_percent:.1f}% NaN values in {dataset_key} {metric_key}")
        
        # Remove NaN values
        data = data[~np.isnan(data)]
        
        if len(data) == 0:
            print(f"Warning: No valid data for {dataset_key} {metric_key}")
            continue
        
        # Create individual figure
        fig, ax = plt.subplots(figsize=(4, 3))
        
        # Plot based on metric type
        if metric_key in ['theta_power', 'speed']:
            # Simple histogram for theta power and speed
            ax.hist(data, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')
            ax.set_xlabel(metric_label, fontsize=8, fontweight='bold')
            ax.set_ylabel('Density', fontsize=8, fontweight='bold')
            ax.tick_params(axis='both', which='major', labelsize=8)
            
            # Limit number of ticks
            ax.xaxis.set_major_locator(ticker.MaxNLocator(4))
            ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
            
        else:
            # Fitter plots for duration and peak power
            f = Fitter(data, distributions=['norm', 'halfnorm', 'lognorm'])
            f.fit()
            
            # Plot histogram
            ax.hist(data, bins=50, density=True, alpha=0.7, color='skyblue', edgecolor='black')
            
            # Plot fitted distributions
            x = np.linspace(data.min(), data.max(), 1000)
            for dist_name, color in zip(['norm', 'halfnorm', 'lognorm'], ['red', 'orange', 'green']):
                if dist_name in f.fitted_param:
                    params = f.fitted_param[dist_name]
                    dist = getattr(stats, dist_name)
                    y = dist.pdf(x, *params)
                    ax.plot(x, y, color=color, label=dist_name, linewidth=2)
            
            ax.set_xlabel(metric_label, fontsize=8, fontweight='bold')
            ax.set_ylabel('Density', fontsize=8, fontweight='bold')
            ax.tick_params(axis='both', which='major', labelsize=8)
            ax.legend(fontsize=8, loc='upper right')
            
            # Limit number of ticks
            ax.xaxis.set_major_locator(ticker.MaxNLocator(4))
            ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
            
            # Extract KS statistics for all distributions
            try:
                summary = f.summary()
                ks_results[dataset_key][metric_key] = {}
                for dist_name in ['norm', 'halfnorm', 'lognorm']:
                    if dist_name in f.fitted_param:
                        ks_results[dataset_key][metric_key][dist_name] = {
                            'sse': summary['sumsquare_error'].get(dist_name, np.nan),
                            'ks_stat': summary.get('ks_stat', {}).get(dist_name, np.nan),
                            'ks_pvalue': summary.get('ks_pvalue', {}).get(dist_name, np.nan),
                            'aic': summary.get('aic', {}).get(dist_name, np.nan),
                            'bic': summary.get('bic', {}).get(dist_name, np.nan)
                        }
                    else:
                        ks_results[dataset_key][metric_key][dist_name] = {
                            'sse': np.nan, 'ks_stat': np.nan, 'ks_pvalue': np.nan,
                            'aic': np.nan, 'bic': np.nan
                        }
            except Exception as e:
                print(f"Error extracting KS statistics for {dataset_key} {metric_key}: {e}")
                ks_results[dataset_key][metric_key] = {
                    'norm': {'sse': np.nan, 'ks_stat': np.nan, 'ks_pvalue': np.nan, 'aic': np.nan, 'bic': np.nan},
                    'halfnorm': {'sse': np.nan, 'ks_stat': np.nan, 'ks_pvalue': np.nan, 'aic': np.nan, 'bic': np.nan},
                    'lognorm': {'sse': np.nan, 'ks_stat': np.nan, 'ks_pvalue': np.nan, 'aic': np.nan, 'bic': np.nan}
                }
        
        # Set title
        ax.set_title(f"{dataset_title}", fontsize=10, fontweight='bold')
        
        # Save individual subplot
        subplot_path_png = os.path.join(INDIVIDUAL_DIR, f"{dataset_key}_{metric_key}.png")
        subplot_path_svg = os.path.join(INDIVIDUAL_DIR, f"{dataset_key}_{metric_key}.svg")
        fig.savefig(subplot_path_png, dpi=300, bbox_inches='tight')
        fig.savefig(subplot_path_svg, bbox_inches='tight')
        
        # Store figure for later combination
        individual_figures.append({
            'fig': fig,
            'ax': ax,
            'row': row,
            'col': col,
            'dataset_key': dataset_key,
            'metric_key': metric_key
        })
        
        plt.close(fig)

# Create combined figure
print("Creating combined figure...")
combined_fig, combined_axes = plt.subplots(4, 3, figsize=(15, 20))

# Add row and column labels
for i, (metric_label, _) in enumerate(METRICS):
    combined_axes[i, 0].set_ylabel(metric_label, fontsize=12, fontweight='bold', rotation=90)

for i, (dataset_title, _) in enumerate(DATASETS):
    combined_axes[0, i].set_title(dataset_title, fontsize=14, fontweight='bold')

# Combine individual figures into the big plot
for item in individual_figures:
    row, col = item['row'], item['col']
    source_ax = item['ax']
    target_ax = combined_axes[row, col]
    
    # Copy all elements from source to target
    for element in source_ax.get_children():
        if hasattr(element, 'get_bbox'):
            bbox = element.get_bbox()
            if bbox is not None:
                # Copy the element to the target axes
                target_ax.add_patch(element)
    
    # Copy plot data
    for line in source_ax.get_lines():
        target_ax.plot(line.get_xdata(), line.get_ydata(), 
                      color=line.get_color(), linewidth=line.get_linewidth(),
                      label=line.get_label())
    
    # Copy histogram data
    for patch in source_ax.patches:
        target_ax.add_patch(patch)
    
    # Copy text elements
    for text in source_ax.texts:
        target_ax.text(text.get_position()[0], text.get_position()[1], 
                      text.get_text(), fontsize=text.get_fontsize(),
                      fontweight=text.get_fontweight())
    
    # Copy axis properties
    target_ax.set_xlabel(source_ax.get_xlabel(), fontsize=8, fontweight='bold')
    target_ax.set_ylabel(source_ax.get_ylabel(), fontsize=8, fontweight='bold')
    target_ax.tick_params(axis='both', which='major', labelsize=8)
    
    # Copy legend if it exists
    if source_ax.get_legend():
        legend = source_ax.get_legend()
        handles, labels = legend.legendHandles, [text.get_text() for text in legend.get_texts()]
        target_ax.legend(handles, labels, fontsize=8, loc='upper right')

# Adjust layout
plt.tight_layout()

# Save combined figure with timestamp
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
combined_fig_path_png = os.path.join(OUT_DIR, f"figure9_combined_{timestamp}.png")
combined_fig_path_svg = os.path.join(OUT_DIR, f"figure9_combined_{timestamp}.svg")

combined_fig.savefig(combined_fig_path_png, dpi=300, bbox_inches='tight')
combined_fig.savefig(combined_fig_path_svg, bbox_inches='tight')

print(f"Combined figure saved as:")
print(f"  PNG: {combined_fig_path_png}")
print(f"  SVG: {combined_fig_path_svg}")

# Save KS results
ks_results_path = os.path.join(OUT_DIR, "figure9_ks_results.json")
with open(ks_results_path, 'w') as f:
    json.dump(ks_results, f, indent=2, default=str)

print(f"KS results saved to: {ks_results_path}")

# Generate and save caption
caption = generate_caption(ks_results)
caption_path = os.path.join(OUT_DIR, "figure9_caption.txt")
with open(caption_path, 'w') as f:
    f.write(caption)

print(f"Caption saved to: {caption_path}")
print("\nCaption:")
print(caption)

plt.close(combined_fig) 