# Figure 9 Regeneration Workflow

This directory contains scripts and resources to regenerate Figure 9 for the SWR event analysis, including data gathering, plotting, and automated caption generation with statistical test results.

## Overview

The workflow consists of three main steps:

1. **Data Gathering**: Extracts mean theta power, mean speed, duration, and peak ripple power for all SWR events from three datasets (ABI Visual Behaviour, ABI Visual Coding, IBL).
2. **Plotting**: Generates a 3x4 grid of histograms and density plots for the datasets, and saves the figure as both PNG and SVG. Also extracts and saves KS test results for use in the figure caption.
3. **Caption Generation**: Produces a .txt file with the figure legend, including the actual KS test results from the data.

---

## IMPORTANT: Environment and Loader Separation

- **Each dataset must be processed in its own Conda environment and with its own script.**
- **ABI Visual Behaviour:**
  - Use `gather_speed_theta_data_abi_visbehave.py`
  - Use the `VisualBehaviorNeuropixelsProjectCache` loader
  - Use the cache at `/space/scratch/allen_visbehave_data`
  - Run in the `allensdk_env` environment
- **ABI Visual Coding:**
  - Use `gather_speed_theta_data_abi_viscoding.py`
  - Use the `EcephysProjectCache` loader
  - Use the cache at `/space/scratch/allen_viscoding_data`
  - Run in the `allensdk_env` environment
- **IBL:**
  - Use `gather_speed_theta_data_ibl.py` (to be provided)
  - Use the appropriate loader and cache for IBL
  - Run in the `ONE_ibl_env` environment

**Do NOT mix up the loader classes or cache directories between ABI Visual Coding and ABI Visual Behaviour.**

---

## Step-by-Step Instructions

### 1. Data Gathering

#### a. ABI Visual Behaviour
```bash
conda activate allensdk_env
python gather_speed_theta_data_abi_visbehave.py
```

#### b. ABI Visual Coding
```bash
conda activate allensdk_env
python gather_speed_theta_data_abi_viscoding.py
```

#### c. IBL
```bash
conda activate ONE_ibl_env
python gather_speed_theta_data_ibl.py
```

Each script will output a `.npz` file in `/space/scratch/SWR_final_pipeline/validation_data_figure9/`.

### 2. Plotting

After all `.npz` files are created, run:
```bash
python plot_figure9.py
```
This will generate the figure (PNG and SVG) and a JSON file with KS test results.

### 3. Caption Generation

After plotting, run the caption generation script (to be provided) to produce the `.txt` file with the actual KS test results inserted into the figure legend.

---

## Requirements
- Python 3.8+
- `allensdk` (for ABI datasets)
- `fitter`, `numpy`, `scipy`, `matplotlib`, `pandas`
- Correct Conda environments for each dataset (see `/Setup/Conda_environments/`)

---

## Troubleshooting
- If you get cache or manifest errors, check that you are using the correct loader and cache directory for the dataset.
- If you get import errors, check that you are in the correct Conda environment.
- If you see missing data warnings, check that the data folders are mounted and accessible.

---

## Contact
For questions, contact the repository maintainer or refer to the documentation in `/SWR_Neuropixels_Detector/README.md`. 