# Figure 9 Workflow

This directory contains the complete workflow for generating Figure 9, which shows SWR event properties across three datasets: ABI Visual Behaviour, ABI Visual Coding, and IBL.

## Quick Start

```bash
# Run the complete workflow
bash run_figure9_workflow.sh

# For debugging: limit to first session per dataset
bash run_figure9_workflow.sh --max_sessions 1

# Skip data gathering and just plot existing data (fast debugging)
bash run_figure9_workflow.sh --skip-gathering
```

## Workflow Steps

1. **Data Gathering**: Collects mean theta power and speed data for SWR events from each dataset
2. **Plotting**: Creates histograms and density plots with statistical fits
3. **KS Test Analysis**: Extracts Kolmogorov-Smirnov test results for distribution fitting
4. **Caption Generation**: Creates figure caption with actual statistical values

## Files

- `gather_speed_theta_data.py`: Vectorized data collection script
- `plot_figure9.py`: Plotting and statistical analysis script
- `run_figure9_workflow.sh`: Complete workflow orchestration
- `theta_1500hz_bandpass_filter.npz`: Theta band filter for LFP analysis

## Command Line Options

### `run_figure9_workflow.sh`

- `--max_sessions N`: Limit processing to first N sessions per dataset (for debugging)
- `--skip-gathering`: Skip data gathering and go straight to plotting (for debugging)
- `--help, -h`: Show help message

### `gather_speed_theta_data.py`

- `--dataset`: Choose dataset (`abi_visbehave`, `abi_viscoding`, `ibl`)
- `--max_sessions N`: Limit number of sessions (for debugging)

## Output Files

- `figure9.png` and `figure9.svg`: The complete figure
- `figure9_ks_results.json`: KS test results for distribution fitting
- `figure9_caption_with_results.txt`: Figure caption with actual statistical values

## Data Processing

The workflow processes putative SWR events that meet the following criteria:
- `power_max_zscore` between 3 and 10
- `sw_peak_power` greater than 1
- No overlap with gamma events (`overlaps_with_gamma == False`)
- No overlap with movement (`overlaps_with_movement == False`)
- Valid start and end times

## Performance Features

- **Vectorized processing**: Uses NumPy operations for fast data processing
- **Lazy loading**: Loads APIs only when needed to avoid dependency conflicts
- **NaN handling**: Robust handling of missing or invalid data
- **Data overwriting**: Automatically removes old data files to ensure fresh results
- **Skip gathering**: Can skip data collection for fast plotting iteration

## Environment Management

The script automatically manages conda environments:
- `allensdk_env` for ABI datasets
- `ONE_ibl_env` for IBL dataset

## Troubleshooting

- **Empty results**: Check that SWR event files exist and meet filtering criteria
- **Missing data**: Ensure LFP and speed data are available for the sessions
- **Environment issues**: Verify conda environments are properly set up
- **Memory issues**: Use `--max_sessions` to limit processing scope

---

## IMPORTANT: Environment and API Separation

- **Each dataset must be processed in its own Conda environment due to API dependencies.**
- **ABI Visual Behaviour:**
  - Uses `VisualBehaviorNeuropixelsProjectCache` from allensdk
  - Uses cache at `/space/scratch/allen_visbehave_data`
  - Run in the `allensdk_env` environment
- **ABI Visual Coding:**
  - Uses `EcephysProjectCache` from allensdk
  - Uses cache at `/space/scratch/allen_viscoding_data`
  - Run in the `allensdk_env` environment
- **IBL:**
  - Uses ONE API to access wheel speed data
  - Run in the `ONE_ibl_env` environment

**The unified script uses lazy loading to avoid dependency conflicts between allensdk and ONE API.**

---

## Scripts

### `gather_speed_theta_data.py`
Unified data gathering script that processes all three datasets using lazy loading.

**Usage:**
```bash
python gather_speed_theta_data.py --dataset [abi_visbehave|abi_viscoding|ibl] [--max_sessions N]
```

**Parameters:**
- `--dataset`: Required. Specify which dataset to process
- `--max_sessions`: Optional. Limit number of sessions for debugging

### `plot_figure9.py`
Generates the figure, extracts KS test results, and creates the caption with statistical values.

**Usage:**
```bash
python plot_figure9.py
```

### `run_figure9_workflow.sh`
Complete automation script that runs all steps in the correct environments.

**Usage:**
```bash
./run_figure9_workflow.sh [--max_sessions N] [--skip-gathering] [--help]
```

**Parameters:**
- `--max_sessions N`: Optional. Limit processing to first N sessions per dataset (for debugging)
- `--skip-gathering`: Optional. Skip data gathering and go straight to plotting (for debugging)
- `--help, -h`: Show help message

---

## Step-by-Step Instructions

### Option 1: Automated Workflow (Recommended)

#### Full Processing
```bash
cd Figures_Tables_and_Technical_Validation/Distribution_v2
./run_figure9_workflow.sh
```

#### Debug Mode (Fast Testing)
```bash
# Process only first session per dataset (very fast)
./run_figure9_workflow.sh --max_sessions 1

# Process first 5 sessions per dataset (moderate speed)
./run_figure9_workflow.sh --max_sessions 5

# Get help
./run_figure9_workflow.sh --help
```

### Option 2: Manual Step-by-Step

#### Step 1: Data Gathering
```bash
# ABI Visual Behaviour
conda activate allensdk_env
python gather_speed_theta_data.py --dataset abi_visbehave --max_sessions 1

# ABI Visual Coding  
conda activate allensdk_env
python gather_speed_theta_data.py --dataset abi_viscoding --max_sessions 1

# IBL
conda activate ONE_ibl_env
python gather_speed_theta_data.py --dataset ibl --max_sessions 1
```

#### Step 2: Plotting and Caption Generation
```bash
conda activate allensdk_env  # or any environment with matplotlib, fitter, etc.
python plot_figure9.py
```

---

## Output Files

After running the workflow, you will have:

- **`figure9.png`** and **`figure9.svg`**: The complete figure with 3x4 grid
- **`figure9_ks_results.json`**: KS test results for statistical analysis
- **`figure9_caption_with_results.txt`**: Caption with actual statistical values inserted
- **`abi_visbehave_swr_theta_speed.npz`**: Processed data for ABI Visual Behaviour
- **`abi_viscoding_swr_theta_speed.npz`**: Processed data for ABI Visual Coding  
- **`ibl_swr_theta_speed.npz`**: Processed data for IBL

---

## Requirements

- Python 3.8+
- **For ABI datasets**: `allensdk` (in `allensdk_env`)
- **For IBL dataset**: `one-api`, `wheel` (in `ONE_ibl_env`)
- **For plotting**: `fitter`, `numpy`, `scipy`, `matplotlib`, `pandas`
- Correct Conda environments for each dataset (see `/Setup/Conda_environments/`)

---

## Technical Details

### Lazy Loading Implementation
The unified script uses lazy loading to avoid dependency conflicts:

```python
def load_abi_visbehave_api():
    from allensdk.brain_observatory.behavior.behavior_project_cache import VisualBehaviorNeuropixelsProjectCache
    return VisualBehaviorNeuropixelsProjectCache

def load_ibl_api():
    from one.api import ONE
    import wheel as wh
    return ONE, wh
```

### Data Processing
- **Theta Power**: Computed using the provided theta filter, Hilbert transform, and z-scoring
- **Speed Data**: Interpolated to 1500 Hz to match LFP sampling rate
- **Event Filtering**: Only events that pass gamma and movement overlap filters are included
- **Statistical Analysis**: Uses Fitter library for distribution fitting and KS test extraction

---

## Debugging and Testing

### Quick Debug Workflow
For fast iteration and bug catching:

```bash
# 1. Test with single session per dataset (fastest)
./run_figure9_workflow.sh --max_sessions 1

# 2. If successful, test with more sessions
./run_figure9_workflow.sh --max_sessions 5

# 3. If still successful, run full processing
./run_figure9_workflow.sh
```

### Individual Dataset Testing
Test specific datasets in isolation:

```bash
# Test only ABI Visual Behaviour
conda activate allensdk_env
python gather_speed_theta_data.py --dataset abi_visbehave --max_sessions 1

# Test only IBL
conda activate ONE_ibl_env  
python gather_speed_theta_data.py --dataset ibl --max_sessions 1
```

---

## Troubleshooting

- **Import errors**: Ensure you're in the correct Conda environment for each dataset
- **Cache errors**: Check that the correct cache directories exist and are accessible
- **Missing data**: Verify that the data folders are mounted and contain the expected files
- **Memory issues**: Use `--max_sessions` to limit processing for testing
- **Long runtime**: Start with `--max_sessions 1` to test the workflow quickly

---

## Contact

For questions, contact the repository maintainer or refer to the documentation in `/SWR_Neuropixels_Detector/README.md`. 