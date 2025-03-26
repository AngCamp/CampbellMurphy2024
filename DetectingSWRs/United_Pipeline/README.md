# SWR Detection Pipeline

This pipeline processes neural recordings from multiple datasets (IBL, Allen Visual Behavior, Allen Visual Coding) to detect Sharp Wave Ripples (SWRs) and related neural events.

## Requirements

- Python 3.8 or higher
- Conda environments for:
  - IBL data: `ONE_ibl_env`
  - Allen Brain data: `allensdk_env`
- Snakemake (install with `pip install snakemake`)

## Setup

1. Clone this repository:
   ```bash
   git clone [repository-url]
   cd United_Pipeline
   ```

2. Make the pipeline script executable:
   ```bash
   chmod +x run_pipeline.sh
   ```

3. Ensure your conda environments have all necessary packages installed.

## Configuration

### Environment Variables

The pipeline uses several environment variables that you can override:

| Variable | Description | Default Value |
|----------|-------------|---------------|
| `DATASET_TO_PROCESS` | The dataset type being processed | `"all"` |
| `RUN_NAME` | Name for this processing run | `"on_the_cluster"` |
| `OUTPUT_DIR` | Directory for output files | `"/space/scratch/SWR_final_pipeline/testing_dir"` |
| `ABI_VISUAL_CODING_SDK_CACHE` | Cache dir for Allen Visual Coding data | `"/space/scratch/allen_viscoding_data"` |
| `ABI_VISUAL_BEHAVIOUR_SDK_CACHE` | Cache dir for Allen Visual Behavior data | `"/space/scratch/allen_visbehave_data"` |
| `IBL_ONEAPI_CACHE` | Cache dir for IBL data | `"/space/scratch/IBL_data_cache"` |

You can override these in two ways:

1. Set them before running the script:
   ```bash
   export OUTPUT_DIR="/my/custom/path"
   ./run_pipeline.sh
   ```

2. Set them inline when running the script:
   ```bash
   OUTPUT_DIR="/my/custom/path" ./run_pipeline.sh
   ```

### Dataset-Specific Settings

Dataset-specific settings are defined in `united_detector_config.yaml`:

- Pool sizes (number of parallel sessions per dataset)
- Detection thresholds
- Output directories
- Dataset-specific parameters

## Running the Pipeline

### Basic Usage

```bash
./run_pipeline.sh [dataset_type] [datasets_to_process] [max_cores]
```

Parameters:
- `dataset_type`: Sets the `DATASET_TO_PROCESS` environment variable (default: "all")
- `datasets_to_process`: Comma-separated list of datasets to process (default: "ibl,abi_visual_behaviour,abi_visual_coding")
- `max_cores`: Maximum number of cores for Snakemake to use (default: 8)

### Examples

```bash
# Run all datasets with default settings
./run_pipeline.sh

# Run only IBL dataset
./run_pipeline.sh ibl "ibl"

# Run Allen datasets with 12 cores
./run_pipeline.sh all "abi_visual_behaviour,abi_visual_coding" 12

# Run everything with a custom output directory
OUTPUT_DIR="/path/to/output" ./run_pipeline.sh
```

## How It Works

1. The pipeline processes datasets sequentially (one after another).
2. Each dataset is processed with a dataset-specific number of parallel sessions:
   - IBL: 2 parallel sessions
   - Allen Visual Behavior: 6 parallel sessions
   - Allen Visual Coding: 6 parallel sessions
3. The sequence is enforced by Snakemake dependencies, while parallelism within datasets is managed by Python's multiprocessing.

## Troubleshooting

### Previously Processed Sessions

If a session directory already exists, the script will skip that session. If you want to reprocess, you need to manually remove the session directories first.

### Memory Issues

If you encounter memory issues, try reducing the pool sizes in the configuration file or running fewer datasets at once.

### Logging

Logs are stored in dataset-specific log files. Check these files if you encounter issues during processing.

## Citation

If you use this pipeline in your research, please cite:
[Citation information]