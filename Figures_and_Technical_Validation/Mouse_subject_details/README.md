# Mouse Demographics Table Generation

This directory contains scripts to generate the mouse demographic summary tables (Tables 1, 2, and 3) for the manuscript. The process involves two steps:
1.  **Collecting Raw Data**: Scripts query dataset APIs and SWR pipeline output folders to gather raw demographic and session information.
2.  **Creating Summary Tables**: A script processes the raw data CSVs to produce the final, formatted summary tables.

## Prerequisites

### 1. Environment Variables
These scripts rely on environment variables to locate the necessary data directories. You must set these before running the scripts. The required variables are the same ones used by the main SWR detection pipeline.

- `OUTPUT_DIR`: The base directory where the SWR pipeline saved its results.
- `ABI_VISUAL_BEHAVIOUR_SDK_CACHE`: The cache directory for the Allen Visual Behaviour dataset.
- `ABI_VISUAL_CODING_SDK_CACHE`: The cache directory for the Allen Visual Coding dataset.

**Example:**
```bash
export OUTPUT_DIR="/space/scratch/SWR_final_pipeline/osf_campbellmurphy2025_v2_final"
export ABI_VISUAL_BEHAVIOUR_SDK_CACHE="/space/scratch/allen_visbehave_data"
export ABI_VISUAL_CODING_SDK_CACHE="/space/scratch/allen_viscoding_data"
```

### 2. Conda Environments
Different scripts require different conda environments, which should already be configured from running the main SWR pipeline.

- **Allen Datasets**: `allensdk_env`
- **IBL Dataset**: `ONE_ibl_env`
- **Summary Script**: Can run in either `allensdk_env` or `ONE_ibl_env` as it only uses `pandas` and `numpy`.

## Step 1: Collect Raw Demographics Data

Run the following scripts to collect the data for each dataset. The output will be saved as CSV files in the `demographics_data/` directory.

### Allen Visual Behaviour (Table 1)
Activate the correct environment and run the script:
```bash
conda activate allensdk_env
python get_abi_visual_behaviour_demographics.py
```
**Output:** `demographics_data/visbehaviour_demographics.csv`

### Allen Visual Coding (Table 2)
Activate the correct environment and run the script:
```bash
conda activate allensdk_env
python get_abi_visual_coding_demographics.py
```
**Output:** `demographics_data/viscoding_demographics.csv`

### IBL Dataset (Table 3)
Activate the correct environment and run the script:
```bash
conda activate ONE_ibl_env
python get_ibl_demographics.py
```
**Output:** `demographics_data/ibl_demographics.csv`


## Step 2: Create Summary Tables

After generating the raw data files, use the `create_summary_table.py` script to generate the final formatted tables for the manuscript. The output tables will be saved in the `demographics_summaries/` directory.

You specify which dataset you want to summarize as a command-line argument.

### Table 1: Allen Visual Behaviour Summary
```bash
python create_summary_table.py abi_visual_behaviour
```
**Output:** `demographics_summaries/visbehaviour_demographics_summary.csv`

### Table 2: Allen Visual Coding Summary
```bash
python create_summary_table.py abi_visual_coding
```
**Output:** `demographics_summaries/viscoding_demographics_summary.csv`

### Table 3: IBL Summary
```bash
python create_summary_table.py ibl
```
**Output:** `demographics_summaries/ibl_demographics_summary.csv` 