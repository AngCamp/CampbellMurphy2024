# Sharp Wave Component Validation

This directory contains comprehensive tools and workflows for validating and visualizing Sharp Wave Ripple (SWR) events from the Neuropixels LFP detection pipeline. The main focus is on exploring SWR characteristics, finding high-quality events, and generating publication-ready visualizations.

## Core Components

### 🔍 **SWRExplorer.py**
The main tool for exploring and analyzing SWR events across multiple datasets. This class provides:

- **Data Loading**: Automatically loads SWR events from Allen Brain Institute (ABI) Visual Behaviour, ABI Visual Coding, and IBL datasets
- **Event Filtering**: Advanced filtering based on sharp wave power, duration, speed, and artifact overlap
- **Event Visualization**: Modular plotting system for individual SWR events with multiple panel options
- **Global Event Analysis**: Tools for analyzing events that occur across multiple probes simultaneously
- **Channel Information**: Integration with AllenSDK for anatomical channel coordinates

**Key Methods:**
- `find_best_events()`: Find high-quality events based on multiple criteria
- `plot_swr_event()`: Generate detailed visualizations of individual SWR events
- `plot_global_swr_event()`: Visualize events that occur across multiple probes
- `filter_events_by_speed()`: Filter events based on running speed
- `get_session_probe_stats()`: Get statistics about events across sessions and probes

### 📊 **Workflow Scripts**

#### `find_best_global_events_workflow.py`
Automated workflow for finding and visualizing the best global SWR events (events that occur across multiple probes):

- **Event Selection**: Finds events with high global peak power (4-6 z-score range)
- **Directionality Analysis**: Identifies anterior-to-posterior or posterior-to-anterior propagating events
- **Quality Filtering**: Applies duration, speed, and probe count filters
- **Spatial Contiguity**: Prioritizes events with spatially contiguous probe participation
- **Output**: Generates PNG and SVG visualizations of top global events

**Key Parameters:**
- `top_n_events`: Number of top events to find (default: 10)
- `direction_filter`: Filter by propagation direction ('anterior', 'posterior', 'directional', 'non_directional')
- `min_probes`/`max_probes`: Range of probe counts per session (default: 3-4)
- `speed_threshold`: Maximum running speed during events (default: 2.0 cm/s)

#### `find_best_events_workflow.py`
Workflow for finding high-quality individual probe events based on various metrics.

#### `plot_specific_swr_event.py`
Script for plotting specific SWR events with customizable parameters:

- **Modular Plotting**: Choose which panels to display (raw LFP, bandpassed signals, power, etc.)
- **Flexible Output**: Generate individual panel plots or combined visualizations
- **High Resolution**: Configurable DPI and output formats (PNG, SVG)

## Generated Figures

### 📁 **figure1d_single_event_components/**
Contains visualizations for Figure 1d components showing:
- **Raw LFP traces**: Pyramidal layer and striatum radiatum layer signals
- **Bandpassed signals**: Ripple band (150-250Hz) and sharp wave band (8-40Hz) filtered signals
- **Power spectra**: Z-scored power envelopes showing event timing
- **Combined visualization**: All components in a single figure

**Files:**
- `swr_event_829_session_1093864136_probe_1094073091_raw_pyramidal_lfp.svg`
- `swr_event_829_session_1093864136_probe_1094073091_raw_s_radiatum_lfp.svg`
- `swr_event_829_session_1093864136_probe_1094073091_bandpassed_signals.svg`
- `swr_event_829_session_1093864136_probe_1094073091_power.svg`
- `swr_event_829_session_1093864136_probe_1094073091.svg` (combined)
- `figure1d_swr_event_829_session_1093864136_probe_1094073091.png` (final figure)

### 📁 **top_global_events/**
Contains visualizations of the best global SWR events (Figure 4 components):

**Event Types:**
- **Individual probe plots**: Shows each participating probe's LFP and power traces
- **Global event summaries**: Combined visualizations showing the event across all probes
- **Directional propagation**: Events showing clear anterior-to-posterior or posterior-to-anterior propagation

**File Naming Convention:**
- `global_event_<rank>_session_<session_id>_id_<event_id>.png/svg` - Individual probe plots
- `global_swr_event_<event_id>_session_<session_id>.png` - Global event summary

**Top Events Generated:**
1. Event 3477 (Session 1093867806) - Anterior propagation
2. Event 478 (Session 1115368723) - Posterior propagation  
3. Event 4414 (Session 1091039376) - Anterior propagation
4. Event 2250 (Session 1086410738) - Anterior propagation
5. Event 1719 (Session 1152811536) - Posterior propagation
6. Event 3397 (Session 1055240613) - Anterior propagation
7. Event 555 (Session 1081429294) - Anterior propagation
8. Event 2244 (Session 1052342277) - Anterior propagation
9. Event 663 (Session 1093867806) - Anterior propagation
10. Event 466 (Session 1104058216) - Anterior propagation

## Publication Figures Generated

This directory generates components for several key publication figures:

1. **Figure 1d**: Single SWR event components showing LFP traces, bandpassed signals, and power spectra
2. **Figure 4**: Global SWR event visualizations showing events propagating across multiple probes
3. **Supplemental Figures**: Various event quality assessments and feature distributions

## Usage Examples

### Basic Event Exploration
```python
from SWRExplorer import SWRExplorer

# Initialize explorer
explorer = SWRExplorer(base_path="/path/to/data")

# List available data
explorer.list_available_data()

# Find best events for a specific session/probe
best_events = explorer.find_best_events(
    dataset="allen_visbehave_swr_murphylab2024",
    session_id="1093864136",
    probe_id="1094073091",
    min_sw_power=1.5,
    min_duration=0.08,
    max_duration=0.1
)
```

### Plot Individual Event
```python
# Plot a specific event with custom parameters
fig = explorer.plot_swr_event(
    events_df=events_df,
    event_idx=829,
    panels_to_plot=['raw_pyramidal_lfp', 'raw_s_radiatum_lfp', 'bandpassed_signals', 'power'],
    window_padding=0.02,
    figsize_mm=(200, 200)
)
```

### Generate Global Event Visualizations
```python
# Run the global events workflow
python find_best_global_events_workflow.py
```

## Data Requirements

- **SWR Pipeline Output**: Requires output from the main SWR detection pipeline
- **AllenSDK**: For channel anatomical coordinates (optional but recommended)
- **LFP Data**: For generating detailed event visualizations
- **Speed Data**: For filtering events by running speed

## Output Formats

All visualizations are generated in both PNG and SVG formats:
- **PNG**: High-resolution raster images for publication
- **SVG**: Vector graphics for further editing in Inkscape or other vector editors

## Quality Control

The workflows include multiple quality control measures:
- **Power thresholds**: Ensure events have sufficient sharp wave power
- **Duration filters**: Remove events that are too short or too long
- **Speed filtering**: Exclude events during high-speed running
- **Directionality validation**: Ensure propagation direction is consistent
- **Spatial contiguity**: Prioritize events with spatially coherent probe participation

This comprehensive validation system ensures that only the highest quality SWR events are selected for publication figures and further analysis. 