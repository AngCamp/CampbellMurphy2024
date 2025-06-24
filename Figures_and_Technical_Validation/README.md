# Figures and Technical Validation

This directory contains all the code, scripts, and outputs used to generate figures and perform technical validation for the Neuropixels LFP Sharp Wave Ripple (SWR) detection pipeline. The contents are organized into specialized subdirectories, each focusing on different aspects of validation and visualization.

## General Workflow

The figure generation process typically follows this pattern:
1. **Data Exploration**: Notebooks or Python scripts are used to sort through the data and identify events of interest
2. **Automated Plotting**: Scripts generate multiple plots and visualizations for selection
3. **Manual Refinement**: Selected plots are modified in Inkscape for publication-quality figures

## Directory Structure

### 📊 **Mouse_subject_details/**
Contains scripts and data for generating subject information summary tables the manuscript. This includes:
- **Scripts**: Python scripts to query dataset APIs and collect subject information from Allen Brain Institute (ABI) Visual Behaviour, ABI Visual Coding, and IBL datasets
- **Data**: Raw subject information CSV files and processed summary tables
- **Outputs**: LaTeX source files and compiled PDF tables for publication
- **Purpose**: Provides comprehensive subject demographics and experimental details for the three datasets used in the study

- **Publication Tables**: Summary tables (Tables 1, 2, and 3).

### 🧠 **Probe_locations_brainrender/**
Contains Jupyter notebooks and visualizations for probe location validation using brainrender:
- **Notebooks**: Interactive notebooks for visualizing probe insertions in 3D brain space for each dataset
- **Images**: Generated figures showing probe placement across different brain regions
- **Purpose**: Validates that probe locations are appropriate for CA1 recording and provides visual documentation of experimental setup

- **Publication Figure**:  The Figure 1a,b,c brian render images are generated in the notebooks. The IBL notebook also contains the code to make the CSD plots from the IBL data, which has higher sampling rate and all channels present.  This is in Figure 5a.

### 🔍 **probe_selection_validation/**
Contains comprehensive validation of the channel selection methodology used in the SWR detection pipeline:
- **Analysis Scripts**: Python scripts for analyzing channel selection metadata and generating validation plots
- **Depth Analysis**: Tools for analyzing probe depth thresholds and channel selection criteria
- **Visualization**: Plots showing ripple and sharp wave features by depth, channel selection metrics
- **Fallback Analysis**: Validation of fallback channel selection methods
- **Generated Figures**: 
  - `ripple_features_by_depth.png` - Shows ripple band power and skewness as a function of depth relative to selected channel
  - `sharp_wave_features_by_depth.png` - Shows sharp wave features by depth relative to ripple channel
  - `sharp_wave_selection_by_metric.png` - Shows channel selection frequency by different metrics (modulation index, net power, etc.)
- **Purpose**: Ensures the automated channel selection process is robust and appropriate for SWR detection
- **Publication Figure**: Figure 5b,c,d Net ripple power and MI selection components are generated here.

### 📈 **Relating_SWR_to_other_data/**
Contains analysis and visualization tools for relating SWR events to other behavioral and neural data:
- **Spike Raster Plots**: Tools for generating spike raster plots around SWR events
- **Behavioral Correlation**: Analysis of SWR events in relation to pupil size and running behavior
- **Visualization Scripts**: Python scripts for creating publication-quality figures
- **Results**: Generated figures showing SWR-behavior relationships
- **Generated Figures**: Multiple spike raster plots showing SWR events with surrounding neural activity
- **Purpose**: Provides insights into the relationship between SWR events and animal behavior/neural activity

### 🌊 **Sankey_plots/**
Contains code for generating Sankey diagrams that visualize the SWR event filtering pipeline:
- **Main Script**: `filteringsankey_real_data.py` - processes event data and applies filtering criteria
- **Event Counting**: `count_total_events.py` - counts SWR events across all datasets
- **Outputs**: PNG and SVG versions of the filtering pipeline visualization
- **Generated Figures**: `v2_real_data_filtering_sankey.png` - Shows the complete filtering pipeline flow
- **Purpose**: Provides a clear visual representation of how events are filtered through the detection pipeline
- **Publication Figure**: Used in the main manuscript to illustrate the filtering process in figure 8.

### ⚡ **Sharp_wave_component_validation/**
Contains comprehensive validation and exploration of sharp wave components and SWR events:
- **Exploration Notebooks**: Interactive notebooks for exploring SWR characteristics and metrics
- **Event Visualization**: Tools for plotting specific SWR events with various signal representations
- **Global Event Analysis**: Analysis of global SWR events across multiple probes
- **Top Event Collections**: Curated collections of high-quality SWR events for validation
- **Generated Figures**: 
  - Multiple global SWR event visualizations showing events across multiple probes
  - Individual SWR event plots with LFP traces, power spectra, and timing
  - Event quality assessments and feature distributions
- **Purpose**: Validates the quality and characteristics of detected SWR events
- **Publication Figure**: Figure 4 (global event visualization) components are generated here, as well as Figure 1d's componenets showing LFP, envelope, power of pyramidal layer (source for ripple band data) and striatum radiatum (source for sharp wave band data).

### ✅ **Validation_notebooks/**
Contains validation notebooks for each dataset to ensure pipeline performance:
- **ABI Visual Behaviour Validation**: `ABI_visbehave_validation.ipynb`
- **ABI Visual Coding Validation**: `ABI_viscoding_validation.ipynb`
- **IBL Validation**: `IBL_validation.ipynb`
- **Additional Scripts**: Supporting Python scripts for validation tasks
- **Generated Figures**: Dataset-specific validation plots including theta power distributions, speed distributions, and ripple feature distributions
- **Purpose**: Provides dataset-specific validation and quality control for the SWR detection pipeline

## Usage Notes

- **Environment Requirements**: Different subdirectories may require different conda environments (see individual README files for details)
- **Data Dependencies**: Most scripts require access to the main SWR pipeline output and dataset caches
- **Output Formats**: Figures are typically generated in both PNG and SVG formats for publication flexibility
- **Validation Workflow**: The validation process follows a systematic approach from probe placement validation to final event quality assessment

## Key Outputs

This directory generates:
- Subject information tables for manuscript publication
- Probe placement validation figures
- Channel selection validation plots
- SWR event quality assessments
- Behavioral correlation analyses
- Pipeline filtering visualizations
- Dataset-specific validation reports

All outputs are designed to support the technical validation and figure generation requirements for the Neuropixels LFP SWR detection manuscript. 