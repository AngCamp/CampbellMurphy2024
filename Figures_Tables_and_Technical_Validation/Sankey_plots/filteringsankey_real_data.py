import plotly.graph_objects as go
import plotly.io as pio
import pandas as pd
import glob
import os
from pathlib import Path

def process_all_datasets(base_path):
    datasets = [
        "allen_visbehave_swr_murphylab2024",
        "allen_viscoding_swr_murphylab2024",
        "ibl_swr_murphylab2024"
    ]
    
    total_stats = {
        'initial_events': 0,
        'passed_sharp_wave': 0,
        'passed_movement': 0,
        'gamma_events': 0,
        'no_gamma_events': 0,
        'power_ranges': {
            '2.5-3': {'gamma': 0, 'no_gamma': 0},
            '3-5': {'gamma': 0, 'no_gamma': 0},
            '5-10': {'gamma': 0, 'no_gamma': 0}
        }
    }
    
    for dataset in datasets:
        dataset_path = os.path.join(base_path, dataset)
        session_folders = glob.glob(os.path.join(dataset_path, "swrs_session_*"))
        
        for session in session_folders:
            event_files = glob.glob(os.path.join(session, "*putative_swr_events.csv.gz"))
            
            for event_file in event_files:
                df = pd.read_csv(event_file, compression='gzip')
                
                # Update statistics
                total_stats['initial_events'] += len(df)
                
                # Sharp wave filter
                sharp_wave_mask = (df['sw_exceeds_threshold'] == True) & (df['power_max_zscore'] > 1)
                total_stats['passed_sharp_wave'] += sharp_wave_mask.sum()
                
                # Movement filter
                movement_mask = ~df['overlaps_with_movement']
                total_stats['passed_movement'] += (sharp_wave_mask & movement_mask).sum()
                
                # Gamma classification
                gamma_mask = df['overlaps_with_gamma']
                total_stats['gamma_events'] += (sharp_wave_mask & movement_mask & gamma_mask).sum()
                total_stats['no_gamma_events'] += (sharp_wave_mask & movement_mask & ~gamma_mask).sum()
                
                # Power ranges
                for power_range, (min_val, max_val) in [
                    ('2.5-3', (2.5, 3)),
                    ('3-5', (3, 5)),
                    ('5-10', (5, 10))
                ]:
                    power_mask = (df['power_max_zscore'] >= min_val) & (df['power_max_zscore'] < max_val)
                    
                    # Gamma events in this power range
                    total_stats['power_ranges'][power_range]['gamma'] += (
                        sharp_wave_mask & movement_mask & gamma_mask & power_mask
                    ).sum()
                    
                    # No gamma events in this power range
                    total_stats['power_ranges'][power_range]['no_gamma'] += (
                        sharp_wave_mask & movement_mask & ~gamma_mask & power_mask
                    ).sum()
    
    return total_stats

# Process the data
#base_path = "yourpath/SWR_final_pipeline/osf_campbellmurphy2025_swr_data"
base_path = "yourpath/SWR_final_pipeline/osf_campbellmurphy2025_v2_final"
stats = process_all_datasets(base_path)

# Format numbers for display
def format_number(num):
    if num >= 1_000_000:
        return f"{num/1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num/1_000:.1f}K"
    return str(num)

# Define node labels with actual data
labels = [
    f"<b>Total Events</b><br><span style='font-size:14px'>{format_number(stats['initial_events'])}</span>",
    "<b>Sharp Wave Filter</b>",
    f"<b>SW Low Power</b><br><span style='font-size:14px'>{format_number(stats['initial_events'] - stats['passed_sharp_wave'])}</span>",
    f"<b>SW High Power</b><br><span style='font-size:14px'>{format_number(stats['passed_sharp_wave'])}</span>", 
    "<b>Movement Filter</b>",
    f"<b>Removed: Movement</b><br><span style='font-size:14px'>{format_number(stats['passed_sharp_wave'] - stats['passed_movement'])}</span>",
    f"<b>Passed: No Movement</b><br><span style='font-size:14px'>{format_number(stats['passed_movement'])}</span>",
    "<b>Gamma Classification</b>",
    f"<b>Gamma Overlap</b><br><span style='font-size:14px'>{format_number(stats['gamma_events'])}</span>",
    f"<b>No Gamma Overlap</b><br><span style='font-size:14px'>{format_number(stats['no_gamma_events'])}</span>",
    f"<b>Gamma: Power 2.5-3</b><br><span style='font-size:14px'>{format_number(stats['power_ranges']['2.5-3']['gamma'])}</span>",
    f"<b>Gamma: Power 3-5</b><br><span style='font-size:14px'>{format_number(stats['power_ranges']['3-5']['gamma'])}</span>", 
    f"<b>Gamma: Power 5-10</b><br><span style='font-size:14px'>{format_number(stats['power_ranges']['5-10']['gamma'])}</span>",
    f"<b>No Gamma: Power 2.5-3</b><br><span style='font-size:14px'>{format_number(stats['power_ranges']['2.5-3']['no_gamma'])}</span>",
    f"<b>No Gamma: Power 3-5</b><br><span style='font-size:14px'>{format_number(stats['power_ranges']['3-5']['no_gamma'])}</span>",
    f"<b>No Gamma: Power 5-10</b><br><span style='font-size:14px'>{format_number(stats['power_ranges']['5-10']['no_gamma'])}</span>"
]

# Define sources and targets
sources = [
    0,  # Total -> Sharp Wave Filter
    1,  # Sharp Wave Filter -> Removed (low power)
    1,  # Sharp Wave Filter -> Passed (high power)
    3,  # Passed Sharp Wave -> Movement Filter
    4,  # Movement Filter -> Removed (movement)
    4,  # Movement Filter -> Passed (no movement)
    6,  # Passed Movement -> Gamma Classification
    7,  # Gamma Classification -> Gamma Overlap
    7,  # Gamma Classification -> No Gamma Overlap
    8,  # Gamma Overlap -> Power 2.5-3
    8,  # Gamma Overlap -> Power 3-5
    8,  # Gamma Overlap -> Power 5-10
    9,  # No Gamma Overlap -> Power 2.5-3
    9,  # No Gamma Overlap -> Power 3-5
    9   # No Gamma Overlap -> Power 5-10
]

targets = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

values = [
    stats['initial_events'],       # Total -> Filter
    stats['initial_events'] - stats['passed_sharp_wave'], # Filter -> Removed
    stats['passed_sharp_wave'],    # Filter -> Passed
    stats['passed_sharp_wave'],    # Passed -> Movement Filter
    stats['passed_sharp_wave'] - stats['passed_movement'], # Movement Filter -> Removed
    stats['passed_movement'],      # Movement Filter -> Passed
    stats['passed_movement'],      # Passed -> Gamma Classification
    stats['gamma_events'],         # Classification -> Gamma
    stats['no_gamma_events'],      # Classification -> No Gamma
    stats['power_ranges']['2.5-3']['gamma'],  # Gamma -> 2.5-3
    stats['power_ranges']['3-5']['gamma'],    # Gamma -> 3-5
    stats['power_ranges']['5-10']['gamma'],   # Gamma -> 5-10
    stats['power_ranges']['2.5-3']['no_gamma'], # No Gamma -> 2.5-3
    stats['power_ranges']['3-5']['no_gamma'],   # No Gamma -> 3-5
    stats['power_ranges']['5-10']['no_gamma']   # No Gamma -> 5-10
]

# Color scheme
node_colors = [
    "#808080",  # 0: Total Events - Grey
    "#808080",  # 1: Sharp Wave Filter - Grey
    "#FF4444",  # 2: Removed Sharp Wave - Red
    "#20B2AA",  # 3: Passed Sharp Wave - Teal
    "#20B2AA",  # 4: Movement Filter - Teal
    "#FF4444",  # 5: Removed Movement - Red
    "#20B2AA",  # 6: Passed Movement - Teal
    "#20B2AA",  # 7: Gamma Classification - Teal
    "#8A2BE2",  # 8: Gamma Overlap - Purple
    "#20B2AA",  # 9: No Gamma Overlap - Teal
    "#8A2BE2",  # 10: Gamma 2.5-3 - Purple
    "#8A2BE2",  # 11: Gamma 3-5 - Purple
    "#8A2BE2",  # 12: Gamma 5-10 - Purple
    "#20B2AA",  # 13: No Gamma 2.5-3 - Teal
    "#20B2AA",  # 14: No Gamma 3-5 - Teal
    "#20B2AA"   # 15: No Gamma 5-10 - Teal
]

# Link colors
link_colors = [
    "rgba(128, 128, 128, 0.4)",  # Total -> Filter
    "rgba(255, 68, 68, 0.7)",    # Filter -> Removed (red)
    "rgba(32, 178, 170, 0.7)",   # Filter -> Passed (teal)
    "rgba(32, 178, 170, 0.4)",   # Passed -> Movement Filter
    "rgba(255, 68, 68, 0.7)",    # Movement -> Removed (red)
    "rgba(32, 178, 170, 0.7)",   # Movement -> Passed (teal)
    "rgba(32, 178, 170, 0.4)",   # Passed -> Gamma Classification
    "rgba(138, 43, 226, 0.7)",   # Classification -> Gamma (purple)
    "rgba(32, 178, 170, 0.7)",   # Classification -> No Gamma (teal)
    "rgba(138, 43, 226, 0.7)",   # Gamma -> 2.5-3 (purple)
    "rgba(138, 43, 226, 0.7)",   # Gamma -> 3-5 (purple)
    "rgba(138, 43, 226, 0.7)",   # Gamma -> 5-10 (purple)
    "rgba(32, 178, 170, 0.7)",   # No Gamma -> 2.5-3 (teal)
    "rgba(32, 178, 170, 0.7)",   # No Gamma -> 3-5 (teal)
    "rgba(32, 178, 170, 0.7)"    # No Gamma -> 5-10 (teal)
]

# Explicit x and y for each node to clarify flow and keep all nodes visible
x = [0.00, 0.12, 0.24, 0.24, 0.36, 0.48, 0.48, 0.60, 0.72, 0.72, 0.96, 0.96, 0.96, 0.96, 0.96, 0.96]
y = [0.5, 0.5, 0.1, 0.9, 0.5, 0.1, 0.9, 0.5, 0.1, 0.9, 0.1, 0.3, 0.5, 0.7, 0.8, 0.9]

# Create the Sankey diagram
fig = go.Figure(data=[go.Sankey(
    node=dict(
        pad=40,
        thickness=40,
        line=dict(color="black", width=2),
        label=labels,
        color=node_colors,
        x=x,
        y=y
    ),
    link=dict(
        source=sources,
        target=targets,
        value=values,
        color=link_colors
    )
)])

# Enhanced layout
fig.update_layout(
    title=dict(
        text="<b>Sharp Wave Ripple Event Filtering Process (Real Data)</b>",
        x=0.5,
        font=dict(size=24, color="#2c3e50", family="Arial Black")
    ),
    font=dict(size=16, family="Arial", color="#2c3e50"),
    width=1600,
    height=1200,
    paper_bgcolor='white',
    plot_bgcolor='white',
    margin=dict(l=50, r=50, t=80, b=50)
)

# Save as PNG and SVG
fig.write_image("v2_real_data_filtering_sankey.png", width=1600, height=1200, scale=2)
fig.write_image("v2_real_data_filtering_sankey.svg", width=1600, height=1200, scale=2)

# Print summary statistics
print("=" * 70)
print("SHARP WAVE RIPPLE FILTERING PIPELINE SUMMARY (REAL DATA)")
print("=" * 70)
print(f"Initial Events: {stats['initial_events']:,}")
print()
print("STEP 1 - Sharp Wave Power Filter (>1 SD):")
print(f"  ✓ Passed:  {stats['passed_sharp_wave']:,} ({100*stats['passed_sharp_wave']/stats['initial_events']:.1f}%)")
print(f"  ✗ Removed: {stats['initial_events'] - stats['passed_sharp_wave']:,} ({100*(stats['initial_events'] - stats['passed_sharp_wave'])/stats['initial_events']:.1f}%)")
print()
print("STEP 2 - Movement Overlap Removal:")
print(f"  ✓ Passed:  {stats['passed_movement']:,} ({100*stats['passed_movement']/stats['passed_sharp_wave']:.1f}% of previous)")
print(f"  ✗ Removed: {stats['passed_sharp_wave'] - stats['passed_movement']:,} ({100*(stats['passed_sharp_wave'] - stats['passed_movement'])/stats['passed_sharp_wave']:.1f}% of previous)")
print()
print("STEP 3 - Gamma Classification:")
print(f"  📶 Gamma Overlap:    {stats['gamma_events']:,} ({100*stats['gamma_events']/stats['passed_movement']:.1f}%)")
print(f"  📊 No Gamma Overlap: {stats['no_gamma_events']:,} ({100*stats['no_gamma_events']/stats['passed_movement']:.1f}%)")
print()
print("FINAL DISTRIBUTION BY PEAK RIPPLE POWER:")
print("Gamma Events:")
print(f"  • 2.5-3 range: {stats['power_ranges']['2.5-3']['gamma']:,}")
print(f"  • 3-5 range:   {stats['power_ranges']['3-5']['gamma']:,}")  
print(f"  • 5-10 range:  {stats['power_ranges']['5-10']['gamma']:,}")
print("No Gamma Events:")
print(f"  • 2.5-3 range: {stats['power_ranges']['2.5-3']['no_gamma']:,}")
print(f"  • 3-5 range:   {stats['power_ranges']['3-5']['no_gamma']:,}")
print(f"  • 5-10 range:  {stats['power_ranges']['5-10']['no_gamma']:,}")
print()
print(f"Total Final Events: {stats['passed_movement']:,}")
print(f"Overall Retention Rate: {100*stats['passed_movement']/stats['initial_events']:.1f}%")
print("=" * 70) 