# Data Usage Tutorials

This folder contains tutorials for understanding and analyzing Sharp Wave Ripple (SWR) data collected from Neuropixels recordings. The tutorials are designed to help researchers work with SWR datasets at different scales and for different analytical purposes.

## Downloading SWR data

The SWR data is published on the [Open Science Framework](https://osf.io/) at the following URL [https://osf.io/9gm6x/](https://osf.io/9gm6x/) (DOI 10.17605/OSF.IO/9GM6X).  Gunzip the data, then to use the notebooks change `SWR_INPUT_DIR` variable, located near the top of either notebook, to the local path, do the same for the cache variables for the Allen SDK or the ONE-api as the case may be.

## Setting up the Allen SDK and the ONE-API

Both the data/development tools for the Allen Brain Institute's data and the International Brain Laboratories data will run with local caches that store the NWB files.  Downloading what is needed as you go to a local cache folder.  The notebooks set the cache using the `CACHE_DIR` varible so these will need to be set as well for you to run the notebooks locally.

## Data Description

These are the columns of the karlsson_filter.csv files.  The columns are in order of appearance, the columns are organized here by their general type of data.  The timing which includes the time of the peak power values.  The power, zscored.  The Sharp-Wave band, from the stratum radiatum.

#### Event Timing information
• 'start_time': The start time of the SWR event  
• 'end_time': The end time of the SWR event  
• 'duration': The duration of the SWR event  
• 'power_peak_time': Time of peak power within the SWR event  

#### Power Z-scores
• 'power_max_zscore': Maximum z-score of power during the event  
• 'power_median_zscore': Median z-score of power during the event  
• 'power_mean_zscore': Mean z-score of power during the event  
• 'power_min_zscore': Minimum z-score of power during the event  
• 'power_90th_percentile': 90th percentile of power z-score during the event  

#### Sharp Wave Metrics
• 'sw_exceeds_threshold': Boolean indicating if sharp wave exceeds detection threshold  
• 'sw_peak_power': Peak power of the sharp wave component  
• 'sw_peak_time': Time of peak sharp wave power  

#### Sharp Wave-Ripple Coupling
• 'sw_ripple_plv': Phase-locking value between sharp wave and ripple components  
• 'sw_ripple_mi': Modulation index between sharp wave and ripple  
• 'sw_ripple_clcorr': Cross-correlation between sharp wave and ripple  

#### Envelope Metrics
• 'envelope_peak_time': Time of peak envelope amplitude  
• 'envelope_max_thresh': Maximum envelope threshold value  
• 'envelope_mean_zscore': Mean z-score of envelope amplitude  
• 'envelope_median_zscore': Median z-score of envelope amplitude  
• 'envelope_max_zscore': Maximum z-score of envelope amplitude  
• 'envelope_min_zscore': Minimum z-score of envelope amplitude  
• 'envelope_area': Area under the envelope curve  
• 'envelope_total_energy': Total energy of the envelope signal  
• 'envelope_90th_percentile': 90th percentile of envelope amplitude  

#### Gamma Overlap
• 'overlaps_with_gamma': Boolean indicating if SWR overlaps with gamma oscillations  
• 'gamma_overlap_percent': Percentage of SWR duration that overlaps with gamma  

#### Movement Overlap
• 'overlaps_with_movement': Boolean indicating if SWR overlaps with movement periods  
• 'movement_overlap_percent': Percentage of SWR duration that overlaps with movement  


## Notebooks Overview

### 1. `swrs_allen_visual_behaviour.ipynb` - Single Session Data Alignment

**Purpose**: This notebook demonstrates how to align and analyze SWR data at the **single session level**.

**Key Features**:
- Shows how to load and organize SWR data for individual recording sessions
- Demonstrates data alignment techniques for single-session analyses
- Provides examples of session-specific SWR event visualization and analysis
- Useful for understanding the structure of SWR data and for pilot analyses

**Use Case**: When you need to examine SWR events within a specific recording session, understand the data structure, or perform preliminary analyses on individual sessions.

### 2. `choosing_event_thresholds.ipynb` - Threshold Testing and Dataset-Level Analyses

**Purpose**: This notebook establishes methods for threshold testing and provides code infrastructure for performing analyses across **entire datasets**.

**Key Features**:
- **Threshold Selection Methods**: Demonstrates how to evaluate and choose appropriate thresholds for SWR detection
- **Dataset-Level Infrastructure**: Provides the `SharpWaveComponentPlotter` object for managing data across multiple sessions and datasets
- **Cross-Dataset Analysis**: Enables comparisons and analyses across entire datasets (e.g., Allen Visual Behavior, Allen Visual Coding, IBL)


**Use Case**: When you need to:
- Test different threshold parameters for SWR detection
- Perform analyses across multiple sessions or datasets
- Build infrastructure for incorporating spiking and behavioral data
- Conduct formal hypothesis testing with proper statistical controls

## Workflow Progression

1. **Start with `swrs_allen_visual_behaviour.ipynb`** to understand the data structure and perform single-session analyses
2. **Progress to `Chosing_Event_thresholds.ipynb`** when you need to:
   - Test threshold parameters systematically
   - Analyze data across multiple sessions
   - Build infrastructure for larger-scale analyses

## Data Infrastructure

The `SharpWaveComponentPlotter` object in the threshold selection notebook provides:
- Hierarchical data organization (dataset → session → probe → event_type)
- Scalable architecture for incorporating additional data types
- Foundation for hypothesis testing with GLMs
- Support for cross-dataset comparisons

## Statistical Considerations

While these notebooks focus on exploratory data analysis (EDA), any formal hypothesis testing should include:
- Bootstrapping for confidence intervals
- Multiple comparisons correction
- Cross-validation
- Effect size reporting

## Getting Started

1. Ensure you have the required dependencies installed
2. Familiarize yourself with the data structure using the single-session notebook
3. Use the threshold selection notebook to establish your analysis pipeline
4. Extend the infrastructure for your specific research questions

## Data Sources

These tutorials work with SWR data from:
- Allen Institute Visual Behavior dataset
- Allen Institute Visual Coding dataset  
- International Brain Laboratory (IBL) dataset 