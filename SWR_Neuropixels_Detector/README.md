# Mouse Hippocampal Sharp Wave Ripple Dataset Curated From Public Neuropixels Datasets

This pipeline processes Neuropixels electrophysiology data from different sources (IBL, Allen Brain Institute Visual Behaviour, Allen Brain Institute Visual Coding) to detect Sharp-Wave Ripples (SWRs).

## Overview
![](Images/Workflow_with_scripts1.png)
## Description
A repo showcasing how to process and analyze Neuropixels LFP from the two largest publicly available datasets: the ABI Visual Behaviour and the IBL dataset.  Currently both the IBL and ABI datasets are missing NWB files on DANDI archive.  Once they are done I'd like to switch to a single pipeline that processes all of them but right now we have to use the individual APIs to get the full datasets.

The pipeline performs the following main steps:

1.  **Data Loading:** Uses dataset-specific loaders (`IBL_loader.py`, `ABI_visual_behaviour_loader.py`, `ABI_visual_coding_loader.py`) built upon a `BaseLoader` class to access session data and LFP signals.
2.  **Channel Selection:** Identifies channels within the CA1 region of the hippocampus and selects optimal channels for ripple and sharp-wave component analysis based on signal properties (power, phase-amplitude coupling).
3.  **Preprocessing:** Resamples LFP data to a common frequency (default 1500 Hz) and applies bandpass filtering (ripple band, sharp-wave component band, gamma band).
4.  **Putative Event Detection:** Detects candidate ripple events using the Karlsson method (`ripple_detection` library) on the selected ripple channel.
5.  **Artifact Detection:** Identifies periods potentially contaminated by gamma bursts or movement artifacts using control channels.
6.  **Event Filtering:** Annotates putative ripple events with information about sharp-wave components and overlap with artifacts. Applies filtering based on thresholds defined in the configuration file.
7.  **Global Event Consolidation:** Merges filtered events across multiple probes within a session that meet specific criteria (e.g., minimum SW power, minimum number of participating probes, minimum CA1 unit counts) to identify session-wide SWR events.
8.  **Cache Cleanup (Optional):** Provides an option to remove cached data downloaded by the respective dataset APIs for specific sessions.

## Project Structure

```
SWR_Neuropixels_Detector/
├── ABI_visual_behaviour_loader.py  # Loader for Allen Visual Behaviour dataset
├── ABI_visual_coding_loader.py     # Loader for Allen Visual Coding dataset
├── Filters/                          # Filter coefficient files (.mat) and design info
│   ├── README.md                     # Filter design documentation
│   └── *.mat                         # Filter files
├── IBL_loader.py                   # Loader for IBL dataset
├── README.md                       # This file
├── Setup/                            # Environment setup information
│   └── README.md
├── swr_neuropixels_collection_core.py # Core functions (BaseLoader, processing, detection algorithms)
├── swr_neuropixels_detector_main.py # Main script orchestrating the pipeline
├── run_pipeline.sh                 # Bash script to run the pipeline
├── united_detector_config.yaml     # Example configuration file
└── session_lists/                  # Directory for lists of session IDs to process
    └── example_session_list.txt
```

## Setup

Please refer to `Setup/README.md` for instructions on setting up the necessary Conda environment.

## Configuration

The pipeline behavior is controlled by two main mechanisms:

1.  **Path Definitions (in `run_pipeline.sh`)**: 
    *   Key directory paths (output directory, dataset cache locations) are defined as variables at the **top of the `run_pipeline.sh` script**. 
    *   **Users should edit these variables directly** in the script to match their system's storage locations.
    *   These paths, if set in the script, will be passed as command-line arguments to the main Python script and will **override** any corresponding paths set in the YAML configuration file.

2.  **Parameters and Flags (YAML Config File)**:
    *   A YAML configuration file (default: `united_detector_config.yaml`) controls all other parameters, including:
        *   Dataset selection (`run_details.dataset_to_process`)
        *   Parallel processing settings (`run_details.max_workers`)
        *   Default behavior flags (`flags.run_putative`, `flags.save_lfp`, etc. - overridden by CLI flags)
        *   Filtering thresholds, channel selection criteria, region definitions, ripple detection parameters, global event consolidation rules, etc.
    *   Users **should review and adjust** the parameters within this file (especially thresholds under `artifact_detection_params`, `global_event_params`, etc.) to suit their specific analysis needs.
    *   The config file path can be specified using the `-c` flag when running `run_pipeline.sh`.

### Config File Structure (`united_detector_config.yaml`)

Key sections in the YAML configuration file include:

*   `paths`: Can specify default locations for directories (logs, SWR results, LFP data, session lists, cache directories) and filter files. Note: Directory paths here are **overridden** if set at the top of `run_pipeline.sh`.
*   `run_details`: Defines the dataset to process (`dataset_to_process`), number of parallel workers (`max_workers`), multiprocessing start method, and an optional run name.
*   `flags`: Controls which stages of the pipeline are executed by default (`run_putative`, `run_filter`, `run_global`, `save_lfp`, `save_channel_metadata`). Can be overridden by command-line flags in `run_pipeline.sh`.
*   `sampling_rates`: Target sampling frequency (`target_fs`).
*   `region_definitions`: Brain region acronyms for hippocampus and control areas.
*   `channel_selection`: Parameters for ripple and sharp-wave channel selection.
*   `ripple_detection_params`: Parameters for the Karlsson ripple detector.
*   `artifact_detection_params`: Parameters for gamma burst and movement artifact detection and filtering thresholds.
*   `global_event_params`: Parameters for merging probe events into global events (merge window, minimum probe count, SW power threshold, minimum CA1 units).


## Running the Pipeline

### Important: Run from the SWR_Neuropixels_Detector directory

All scripts and commands should be executed from the `SWR_Neuropixels_Detector` directory. The pipeline is designed to be run from this location and uses relative paths for accessing configuration files, filter files, and session lists.

```bash
# Navigate to the SWR_Neuropixels_Detector directory
cd /path/to/SWR_Neuropixels_Detector

# Make the script executable (if needed)
chmod +x run_pipeline.sh

# Run with default settings (all datasets, all stages)
./run_pipeline.sh 
```

### Important Configuration Variables

The most important configuration variables are now placed at the top of the `run_pipeline.sh` script for easy access:

```bash
# Output directory for all results
export OUTPUT_DIR=${OUTPUT_DIR:-"/space/scratch/SWR_final_pipeline/muckingabout"}

# Cache directories for datasets (where raw data is stored/downloaded)
export ABI_VISUAL_CODING_SDK_CACHE=${ABI_VISUAL_CODING_SDK_CACHE:-"/space/scratch/allen_viscoding_data"}
export ABI_VISUAL_BEHAVIOUR_SDK_CACHE=${ABI_VISUAL_BEHAVIOUR_SDK_CACHE:-"/space/scratch/allen_visbehave_data"}
export IBL_ONEAPI_CACHE=${IBL_ONEAPI_CACHE:-"/space/scratch/IBL_data_cache"}
```

You should edit these variables directly in the script to match your system's storage locations before running the pipeline.

### Dataset Selection

There are two ways to specify which datasets to process:

1. **Using positional arguments (preferred):**
   ```bash
   # Process all datasets
   ./run_pipeline.sh all
   
   # Process only the IBL dataset
   ./run_pipeline.sh subset ibl
   
   # Process multiple specific datasets
   ./run_pipeline.sh subset "ibl,abi_visual_behaviour"
   
   # Debug mode for a specific dataset
   ./run_pipeline.sh debug ibl
   ```

2. **Using environment variables:**
   ```bash
   # Process only the IBL dataset
   DATASET_TO_PROCESS=ibl ./run_pipeline.sh
   
   # Process multiple datasets
   DATASET_TO_PROCESS="ibl,abi_visual_behaviour" ./run_pipeline.sh
   ```

### Command-Line Flags

The `run_pipeline.sh` script accepts the following command-line flags, which can be combined with the dataset selection:

| Flag | Long Form | Description |
|------|-----------|-------------|
| `-h` | `--help` | Display the help message and exit |
| `-c FILE` | `--config FILE` | Specify a custom configuration YAML file (default: united_detector_config.yaml) |
| `-fg` | `--find-global` | Run global event detection using existing probe events (skip probe processing) |
| `-s` | `--save-lfp` | Enable saving of LFP data (overrides config) |
| `-m` | `--save-channel-metadata` | Enable saving of channel selection metadata |
| `-o` | `--overwrite-existing` | Overwrite existing session output folders |
| `-X` | `--cleanup-after` | Clean up cache after processing each session |
| `-d` | `--debug` | Enable debug mode (debugpy listening on port 5678) |

**Notes:**
- The `-fg` flag is useful when you want to rerun global event detection using existing probe events without reprocessing the probes. This is helpful when you want to try different global event parameters without redoing the computationally expensive probe processing.
- The save flags (`-s`, `-m`) and debug flag (`-d`) act as overrides/triggers and can be combined with other flags.
- Before running the script, ensure the correct conda environment is activated (e.g., `ONE_ibl_env` for IBL data, `allensdk_env` for Allen Brain Institute data).

### Environment Variables

You can also control other aspects of the pipeline behavior by setting environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `OUTPUT_DIR` | Base output directory for results | Specified in config file |
| `CONFIG_PATH` | Custom path to configuration YAML file | ./united_detector_config.yaml |

### Examples

```bash
# Display help message
./run_pipeline.sh -h

# Run all datasets with all processing stages
./run_pipeline.sh

# Run only the IBL dataset
./run_pipeline.sh subset ibl

# Run with LFP saving enabled
./run_pipeline.sh subset ibl -s

# Rerun global event detection using existing probe events
# we wanted this so that the user had to explicitly delete files or make 
# a copy of the session folders without the gobal events 
./run_pipeline.sh subset abi_visual_behaviour -fg -o

# Run with metadata saving and overwrite existing output
./run_pipeline.sh subset "ibl,abi_visual_behaviour" -m -o

# Debug the IBL dataset
./run_pipeline.sh subset ibl -d
```

Remember that you need to activate the correct conda environment for your dataset:
- For IBL data: `conda activate ONE_ibl_env`
- For Allen data (both visual behavior and coding): `conda activate allensdk_env`

## Output Files

Outputs are saved in the directory specified by `paths.swr_output_dir` in the config file, organized into session-specific subfolders (e.g., `swrs_session_<session_id>/`).

**Key Output Files per Session:**

*   `probe_metadata.csv.gz`: Aggregated metadata for all probes in the session (channel counts, unit counts, position ranges).
*   `probe_<probe_id>_channel_selection_metadata.json.gz` (Optional, if `-m` or `--save-channel-metadata` used): Detailed metrics used for selecting the ripple and sharp-wave channels for a specific probe (power, skew, coupling values for evaluated channels).
*   `probe_<probe_id>_karlsson_detector_events.csv.gz`: Detected SWR events for a specific probe. 
    *   If only the putative stage (`-p`) is run, this contains raw detected events.
    *   If the filter stage (`-f`) is run, this file is **overwritten** with events annotated with artifact overlap percentages, sharp-wave component metrics, and filtered based on config thresholds.
*   `probe_<probe_id>_channel_<chan_id>_gamma_band_events.csv.gz`: Detected gamma burst events on the selected ripple channel (used for artifact filtering).
*   `probe_<probe_id>_channel_controlid_<control_chan_id>_movement_artifacts.csv.gz`: Detected high-amplitude events on control channels (used for movement artifact filtering).
*   `session_<session_id>_global_swrs_<label>.csv.gz`: Consolidated global SWR events for the session, created after merging and filtering probe events based on `global_event_params` in the config. The `<label>` is also defined in the config (`global_event_params.global_rip_label`).

**Optional LFP Output (if `-s` or `--save-lfp` used or `flags.save_lfp: true` in config):**

*   Raw LFP data for selected channels (ripple, sharp-wave, controls) can be saved in `.npz` format to the directory specified by `paths.lfp_output_dir`.

## Logging

Logs are saved to the directory specified by `paths.log_dir` (default: `./logs/`). A main `pipeline_run.log` file captures the overall process and errors.

## Troubleshooting

### Pipeline Hanging

If the pipeline seems to hang when a session fails, this may be due to the multiprocessing error handling. When a worker process encounters an exception, the pipeline may not properly terminate or continue processing other sessions. To address this issue:

1. **Check log files:** Look at the logs in the `LOG_DIR` to identify which session(s) may be causing the issue.

2. **Run with fewer parallel sessions:** Reduce the `pool_size` value in your configuration file to process fewer sessions in parallel, which can help isolate problematic sessions.

3. **Run specific problem sessions individually:** If you identify problematic sessions, you can run them individually with additional debugging:
   ```bash
   ./run_pipeline.sh subset ibl -d
   ```

4. **Memory issues:** Some sessions may fail due to memory constraints. If a worker process exceeds available memory, it may hang rather than crash properly. Consider increasing system memory or reducing the number of parallel workers.

5. **One-by-one processing:** If you continue to experience hanging issues, you can modify the pipeline to process sessions one by one by setting `pool_size: 1` in your configuration.

### Empty Session Directories

The pipeline automatically cleans up empty session directories that might result from failed processing.

### Memory Issues

If you encounter memory issues, try adjusting the pool sizes in the configuration file:

```yaml
pool_sizes:
  abi_visual_behaviour: 6  # Reduce this number if needed
  abi_visual_coding: 6     # Reduce this number if needed
  ibl: 2                   # Reduce this number if needed
```

### Missing Cache Directories

If the pipeline fails to find data in the cache directories, ensure that:
1. The paths at the top of `run_pipeline.sh` are correctly set to your local cache directories
2. You have the necessary permissions to access these directories
3. The cache directories contain the expected data structure for each dataset

## Debugging

For developers who want to step through the Python code:

1.  **Install `debugpy`:** Ensure the `debugpy` package is installed in the Conda environment used by the pipeline:
    ```bash
    conda activate <your_environment_name>
    pip install debugpy
    ```

2.  **Run with Debug Flag:** Execute the `run_pipeline.sh` script with the `-d` flag. This tells the main Python script to start a debugpy listener on port 5678 and wait for a debugger to attach before proceeding.
    ```bash
    ./run_pipeline.sh -d [other flags...] 
    # Example: Debug putative stage for one session
    # ./run_pipeline.sh -d -p 
    ```
    The script will print messages indicating it's waiting for attachment.

3.  **Attach VS Code Debugger:** Use a VS Code launch configuration to attach to the running script. Create a `.vscode/launch.json` file in the root of your project directory (`SWR_Neuropixels_Detector`) with the following content:

    ```json
    {
        "version": "0.2.0",
        "configurations": [
            {
                "name": "Python: Attach to Pipeline",
                "type": "python",
                "request": "attach",
                "connect": {
                    "host": "localhost", // Or the remote host if running remotely
                    "port": 5678
                },
                "pathMappings": [
                    {
                        "localRoot": "${workspaceFolder}", // Maps your VS Code workspace root
                        "remoteRoot": "." // Maps to the directory where the script is running
                    }
                    // Add more mappings if your source code location differs significantly from the execution location (e.g., Docker)
                ],
                "justMyCode": true // Set to false to step into library code
            }
        ]
    }
    ```
    *   Adjust `"host": "localhost"` if the script is running on a remote machine you are connected to (use the remote hostname or IP).
    *   Adjust `pathMappings` if necessary, especially if using remote connections or containers.
    *   Select the "Python: Attach to Pipeline" configuration in the VS Code Run and Debug panel (usually Ctrl+Shift+D) and press F5 (or the green play button).

4.  **Debug:** Once attached, you can set breakpoints, step through the code, inspect variables, etc., in `swr_neuropixels_detector_main.py` and any modules it calls (like `swr_neuropixels_collection_core.py` or the loaders).

    **Note:** Debugging multiprocessing code can be complex. The current setup attaches to the main process *before* it spawns worker processes. To debug worker processes, more advanced techniques (like having workers start their own debugpy listeners on different ports) would be needed.

## For Developers

This section provides a brief overview of the code structure and logic for those wishing to modify or extend the pipeline.

### Core Components

*   **`run_pipeline.sh`**: Entry point script. Parses command-line arguments (using `getopts` for single-letter flags), sets up the environment (activates Conda), determines which stages/flags to pass to Python (using descriptive long-form arguments), and executes the main Python script.
*   **`swr_neuropixels_detector_main.py`**: Main Python script. 
    *   Parses descriptive arguments passed from the bash script (e.g., `--run-putative`, `--cleanup-cache`, `--save-lfp`, `--save-channel-metadata`, `--config`).
    *   Loads the configuration file.
    *   Sets up logging.
    *   Loads the list of session IDs to process.
    *   Determines final flag settings (CLI args override config flags).
    *   Sets up multiprocessing pool based on `max_workers` and `multiprocessing_start_method` from config.
    *   If cleanup is requested, calls `cleanup_session_cache` for each session (potentially in parallel).
    *   If processing stages are requested, calls `process_session` for each session (potentially in parallel).
*   **`swr_neuropixels_collection_core.py`**: Contains the core logic and shared functions.
    *   `BaseLoader`: Abstract base class defining the interface for dataset-specific loaders. Includes common methods like `resample_signal` and `select_sharpwave_channel`.
    *   `process_session`: The main function executed for each session. It orchestrates:
        1.  Setting up the appropriate `BaseLoader` instance.
        2.  Checking for probes with CA1.
        3.  Generating probe metadata (if enabled).
        4.  Looping through CA1 probes:
            *   Calling `loader.process_probe` to get LFP, select channels, etc. (Putative Stage).
            *   Performing artifact detection.
            *   Running Karlsson SWR detection.
            *   Saving putative events and metadata (conditionally based on `save_channel_metadata` flag).
            *   Optionally saving LFP data (conditionally based on `save_lfp` flag).
            *   Performing event filtering (Filter Stage) - loading artifacts, adding SW info, applying thresholds, overwriting event file.
        5.  Performing global event detection (Global Stage) - loading filtered probe events, applying global criteria, saving global event file.
    *   Other utility functions for filtering, event detection, etc.
*   **`*_loader.py` (e.g., `IBL_loader.py`)**: Dataset-specific implementations inheriting from `BaseLoader`. They handle the specifics of interacting with each dataset's API, file structure, and metadata conventions to fulfill the methods required by `BaseLoader` (e.g., `set_up`, `get_probes_with_ca1`, `process_probe`, `cleanup_cache`, `get_probe_channel_info`, etc.).
*   **`united_detector_config.yaml`**: Central configuration controlling parameters, paths, and default flags (e.g., `flags.save_lfp`, `flags.save_channel_metadata`).

### Execution Flow

1.  `run_pipeline.sh` parses single-letter flags (e.g., `-p`, `-s`, `-C`, `-m`).
2.  `run_pipeline.sh` assumes the correct conda environment (e.g., `ONE_ibl_env` or `allensdk_env`) is already activated.
3.  `run_pipeline.sh` executes `swr_neuropixels_detector_main.py`, passing descriptive arguments corresponding to the set flags (e.g., `--run-putative`, `--save-lfp`, `--cleanup-cache`, `--save-channel-metadata`).
4.  `swr_neuropixels_detector_main.py` loads the config, session list, and sets up logging.
5.  It determines the final state of flags (CLI args override config flags).
6.  It potentially runs `cleanup_session_cache` in parallel/sequentially for each session ID.
7.  It potentially runs `process_session` in parallel/sequentially for each session ID, passing the config dictionary (which includes the final flag states) to it.
8.  `process_session` uses the flags within the passed config (`config['flags']`) to determine which internal stages (putative, filter, global, save LFP, save channel metadata) to execute for that specific session.

### Adding a New Dataset

1.  Create a new loader class (e.g., `MyDataset_loader.py`) that inherits from `BaseLoader`.
2.  Implement all abstract methods defined in `BaseLoader` (`set_up`, `has_ca1_channels`, `get_probes_with_ca1`, `process_probe`, `global_events_probe_info`, `cleanup`, `cleanup_cache`, `get_all_probe_ids`, `get_probe_channel_info`, `get_probe_unit_info`). Ensure `process_probe` returns a dictionary with the keys expected by `process_session`.
3.  Update the `BaseLoader.create` factory method in `swr_neuropixels_collection_core.py` to recognize and instantiate your new loader based on a `dataset_to_process` string.
4.  Add necessary configuration parameters for your dataset (e.g., cache paths) to `united_detector_config.yaml`.
5.  Add any specific dependencies to the Conda environment.

## Requirements

- Python 3.8 or higher
- Conda environments for:
  - IBL data: `ONE_ibl_env`
  - Allen Brain data: `allensdk_env`


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

## Citation

If you use this pipeline in your research, please cite:
[Citation information]