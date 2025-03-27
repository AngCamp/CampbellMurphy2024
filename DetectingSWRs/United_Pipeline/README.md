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
| `RUN_NAME` | Name for this processing run | `"on_the_cluster"` |
| `OUTPUT_DIR` | Directory for output files | `"/space/scratch/SWR_final_pipeline/testing_dir"` |
| `LOG_DIR` | Directory for log files | `"${OUTPUT_DIR}/logs"` |
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
# Run all datasets with default settings
./run_pipeline.sh

# Explicitly run all datasets
./run_pipeline.sh all
```

### Running Specific Datasets

```bash
# Run only the IBL dataset
./run_pipeline.sh subset ibl

# Run only the Allen Visual Behavior dataset
./run_pipeline.sh subset abi_visual_behaviour

# Run multiple specific datasets
./run_pipeline.sh subset "ibl,abi_visual_coding"
```

### Additional Parameters

The pipeline automatically handles resource allocation based on the settings in the configuration file. For advanced usage scenarios, additional parameters can be passed when running the script. Refer to the help information for details:

```bash
./run_pipeline.sh --help
```

### Getting Help

```bash
./run_pipeline.sh --help
```

## How It Works

1. The pipeline processes datasets sequentially (one after another).
2. Each dataset is processed with a dataset-specific number of parallel sessions:
   - IBL: 2 parallel sessions
   - Allen Visual Behavior: 6 parallel sessions
   - Allen Visual Coding: 6 parallel sessions
3. The sequence is enforced by Snakemake dependencies, while parallelism within datasets is managed by Python's multiprocessing.

## Output Structure

```
OUTPUT_DIR/
├── ibl_swr_murphylab2024/             # IBL dataset results
│   └── swrs_session_[session_id]/     # Session-specific results
├── allen_visbehave_swr_murphylab2024/ # Allen Visual Behavior results
├── allen_viscoding_swr_murphylab2024/ # Allen Visual Coding results
├── logs/                              # Log files
└── results/                           # Snakemake results and markers
```

## Troubleshooting

### Empty Session Directories

The pipeline automatically cleans up empty session directories and logs them.

### Memory Issues

If you encounter memory issues, try adjusting the pool sizes in the configuration file:

```yaml
pool_sizes:
  abi_visual_behaviour: 6  # Reduce this number if needed
  abi_visual_coding: 6     # Reduce this number if needed
  ibl: 2                   # Reduce this number if needed
```

### Log Files

Check the log files in the `LOG_DIR` for detailed information about each dataset's processing.

## Citation

If you use this pipeline in your research, please cite:
[Citation information]