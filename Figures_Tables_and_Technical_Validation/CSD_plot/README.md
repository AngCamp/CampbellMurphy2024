# Current Source Density (CSD) Analysis

This directory contains code and workflows for computing and visualizing Current Source Density (CSD) from Local Field Potential (LFP) data during Sharp Wave Ripple (SWR) events. The CSD analysis provides insights into the spatial distribution of current sources and sinks in the hippocampus during SWR events.

## Overview

Current Source Density analysis computes the second spatial derivative of the LFP signal across electrode channels, revealing the underlying current sources and sinks in the brain tissue. This is particularly useful for understanding the laminar organization of SWR events in the hippocampus.

## Files and Workflows

### 📊 **IBL_validation.ipynb**
- **Purpose**: Jupyter notebook for generating CSD plots from IBL (International Brain Laboratory) data
- **Publication Figure**: Generates components for Figure 5A in the manuscript
- **Features**: 
  - Uses IBL data which has higher sampling rate and complete channel information
  - Provides validation of CSD methodology across different datasets
  - Generates publication-quality CSD visualizations

### 🔄 **csd_swr_events_workflow.py**
- **Purpose**: Main workflow for computing CSD on individual SWR events from ABI Visual Behaviour dataset
- **Key Features**:
  - Identifies good recordings (probes in middle of CA1 with sufficient buffer)
  - Filters for high-quality SWR events (immobile, duration 75-150ms)
  - Computes CSD for each probe centered around peak global ripple time
  - Applies various smoothing methods (exponential, Gaussian, Savitzky-Golay)
  - Generates individual event CSD plots

### 📈 **csd_trial_average_workflow.py**
- **Purpose**: Modified workflow for computing trial-averaged CSD across multiple events
- **Key Features**:
  - Averages CSD across all events that pass quality thresholds
  - Aligns events to peak ripple time for proper averaging
  - Provides both mean and median averaging options
  - Generates trial-averaged CSD plots for more robust analysis

### 🎛️ **theta_1500hz_bandpass_filter.npz**
- **Purpose**: Pre-computed bandpass filter coefficients for theta frequency band
- **Usage**: Applied to LFP data before CSD computation to focus on relevant frequency components

## Output Directories

### 📁 **csd_results/**
Contains outputs from individual SWR event CSD analysis:
- **csd_data/**: Raw CSD computation results stored as .npz files
- **csd_plots/**: Generated CSD plots in PNG and SVG formats
- **csd_events_summary.json**: Summary metadata for processed events

### 📁 **csd_trial_mean/**
Contains outputs from trial-averaged CSD analysis:
- **csd_data/**: Trial-averaged CSD computation results
- **csd_plots/**: Trial-averaged CSD plots
- **trial_averaged_csd_events_summary.json**: Summary of trial averaging process

## Key Parameters

### CSD Computation Parameters
- `CSD_COMPUTE_DEPTH_RANGE`: 500 microns from pyramidal layer for CSD computation
- `CSD_PLOT_DEPTH_RANGE`: (-200, 100) microns from pyramidal layer for plotting
- `MIN_CHANNELS_BELOW_PYRAMIDAL`: 10 channels below pyramidal layer
- `MIN_CHANNELS_ABOVE_PYRAMIDAL`: 5 channels above pyramidal layer

### Event Filtering Parameters
- `PUTATIVE_MIN_POWER_MAX_ZSCORE`: 3.0 (minimum ripple power threshold)
- `PUTATIVE_MAX_POWER_MAX_ZSCORE`: 10.0 (maximum ripple power threshold)
- `PUTATIVE_SPEED_THRESHOLD`: 2.0 cm/s (maximum running speed)
- `PUTATIVE_SPEED_EXCLUSION_WINDOW`: 2.0 seconds (speed exclusion window)

### Smoothing Options
- **Exponential smoothing**: Alpha parameters for spatial and temporal smoothing
- **Gaussian smoothing**: Sigma and truncate parameters
- **Savitzky-Golay smoothing**: Window length and polynomial order

## CSD Computation Method

The CSD is computed using the second spatial derivative:

```
CSD = (LFP[i+1] - 2*LFP[i] + LFP[i-1]) / (spacing²)
```

Where:
- `LFP[i]` is the LFP signal at channel i
- `spacing` is the distance between adjacent channels in micrometers

## Usage

### For Individual Events
```bash
python csd_swr_events_workflow.py
```

### For Trial Averaging
```bash
python csd_trial_average_workflow.py
```

### For IBL Validation
Open `IBL_validation.ipynb` in Jupyter and run the cells sequentially.

## Dependencies

- **AllenSDK**: For loading ABI Visual Behaviour data
- **NumPy/SciPy**: For numerical computations and signal processing
- **Matplotlib**: For plotting CSD visualizations
- **Pandas**: For data manipulation and analysis

## Publication Figures

This directory generates components for:
- **Figure 5A**: CSD plots from IBL data showing laminar organization of SWR events
- **Supplemental Figures**: Individual and trial-averaged CSD plots for validation

## Notes

- The workflow automatically selects the best recording sessions based on probe placement and channel density
- CSD plots are generated with proper depth scaling relative to the pyramidal layer
- Multiple smoothing options are available to reduce noise while preserving signal structure
- Trial averaging provides more robust CSD estimates by reducing event-to-event variability 