# Figures and Technical Validation

This directory contains all the code, scripts, and outputs used to generate figures and perform technical validation for the Neuropixels LFP Sharp Wave Ripple (SWR) detection pipeline. The contents are organized into specialized subdirectories, each focusing on different aspects of validation and visualization.

## Directory Structure

### 📊 **Mouse_subject_details/**
Contains scripts and data for generating subject information summary tables (Tables 1, 2, and 3) for the manuscript. This includes:
- **Scripts**: Python scripts to query dataset APIs and collect subject information from Allen Brain Institute (ABI) Visual Behaviour, ABI Visual Coding, and IBL datasets
- **Data**: Raw subject information CSV files and processed summary tables
- **Outputs**: LaTeX source files and compiled PDF tables for publication
- **Purpose**: Provides comprehensive subject demographics and experimental details for the three datasets used in the study

### 🧠 **Probe_locations_brainrender/**
Contains Jupyter notebooks and visualizations for probe location validation using brainrender:
- **Notebooks**: Interactive notebooks for visualizing probe insertions in 3D brain space for each dataset
- **Images**: Generated figures showing probe placement across different brain regions
- **Purpose**: Validates that probe locations are appropriate for CA1 recording and provides visual documentation of experimental setup

### 🔍 **probe_selection_validation/**
Contains comprehensive validation of the channel selection methodology used in the SWR detection pipeline:
- **Analysis Scripts**: Python scripts for analyzing channel selection metadata and generating validation plots
- **Depth Analysis**: Tools for analyzing probe depth thresholds and channel selection criteria
- **Visualization**: Plots showing ripple and sharp wave features by depth, channel selection metrics
- **Fallback Analysis**: Validation of fallback channel selection methods
- **Purpose**: Ensures the automated channel selection process is robust and appropriate for SWR detection

### 📈 **Relating_SWR_to_other_data/**
Contains analysis and visualization tools for relating SWR events to other behavioral and neural data:
- **Spike Raster Plots**: Tools for generating spike raster plots around SWR events
- **Behavioral Correlation**: Analysis of SWR events in relation to pupil size and running behavior
- **Visualization Scripts**: Python scripts for creating publication-quality figures
- **Results**: Generated figures showing SWR-behavior relationships
- **Purpose**: Provides insights into the relationship between SWR events and animal behavior/neural activity

### 🌊 **Sankey_plots/**
Contains code for generating Sankey diagrams that visualize the SWR event filtering pipeline:
- **Main Script**: `filteringsankey_real_data.py` - processes event data and applies filtering criteria
- **Event Counting**: `count_total_events.py` - counts SWR events across all datasets
- **Outputs**: PNG and SVG versions of the filtering pipeline visualization
- **Purpose**: Provides a clear visual representation of how events are filtered through the detection pipeline

### ⚡ **Sharp_wave_component_validation/**
Contains comprehensive validation and exploration of sharp wave components and SWR events:
- **Exploration Notebooks**: Interactive notebooks for exploring SWR characteristics and metrics
- **Event Visualization**: Tools for plotting specific SWR events with various signal representations
- **Global Event Analysis**: Analysis of global SWR events across multiple probes
- **Top Event Collections**: Curated collections of high-quality SWR events for validation
- **Purpose**: Validates the quality and characteristics of detected SWR events

### ✅ **Validation_notebooks/**
Contains validation notebooks for each dataset to ensure pipeline performance:
- **ABI Visual Behaviour Validation**: `ABI_visbehave_validation.ipynb`
- **ABI Visual Coding Validation**: `ABI_viscoding_validation.ipynb`
- **IBL Validation**: `IBL_validation.ipynb`
- **Additional Scripts**: Supporting Python scripts for validation tasks
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