#!/bin/bash

# =====================================================================
# IMPORTANT CONFIGURATION VARIABLES - EDIT THESE FOR YOUR ENVIRONMENT
# =====================================================================

# Output directory for all SWR detection results
# This is where all processed data, events, and metadata will be saved
# Structure: $OUTPUT_DIR/{dataset_name}/{session_id}/...
export OUTPUT_DIR=${OUTPUT_DIR:-"your/path_to/output_directory"}

# Cache directories for datasets - these store raw neurophysiology data downloaded by APIs/SDKs
# Each dataset API maintains its own cache to avoid re-downloading large files
# These directories can grow to hundreds of GB, so ensure sufficient storage space

# Allen Brain Institute Visual Coding dataset cache (AllenSDK)
export ABI_VISUAL_CODING_SDK_CACHE=${ABI_VISUAL_CODING_SDK_CACHE:-"your/path_to/ABI_visual_coding_cache"}

# Allen Brain Institute Visual Behaviour dataset cache (AllenSDK)
export ABI_VISUAL_BEHAVIOUR_SDK_CACHE=${ABI_VISUAL_BEHAVIOUR_SDK_CACHE:-"your/path_to/ABI_visual_behavior_cache"}

# International Brain Laboratory dataset cache (ONE-API)
export IBL_ONEAPI_CACHE=${IBL_ONEAPI_CACHE:-"your/path_to/IBL_data_cache"}

# Run name for tracking pipeline settings across different runs
# Consider including date/time for better organization: "swr_detection_$(date +%Y%m%d_%H%M%S)"
# Detection thresholds from config file are stored with each session's output
export RUN_NAME=${RUN_NAME:-"run_name_here_$(date +%Y%m%d_%H%M%S)"}

# prevents pycache files from being created in working directory
export PYTHONDONTWRITEBYTECODE=1 

# =====================================================================
# Display help information
show_help() {
  echo "SWR Neuropixels Detector Pipeline"
  echo "======================================================"
  echo "Usage: ./run_pipeline.sh [command] [datasets] [options]"
  echo ""
  echo "Commands:"
  echo "  all                     Process all datasets (default if no command provided)"
  echo "  subset DATASETS         Process only specified datasets"
  echo "  debug DATASET           Run in debug mode for a specific dataset"
  echo ""
  echo "Parameters:"
  echo "  DATASETS                Comma-separated list of datasets to process"
  echo "                          Valid values: ibl, abi_visual_behaviour, abi_visual_coding"
  echo "                          Example: 'ibl' or 'ibl,abi_visual_behaviour'"
  echo ""
  echo "Options:"
  echo "  -h, --help                      Show this help message and exit"
  echo "  -c, --config FILE               Specify a custom configuration YAML file"
  echo "                                  (default: united_detector_config.yaml)"
  echo "  -fg, --find-global             Run global event detection using existing probe events (skip probe processing)"
  echo "  -s, --save-lfp, --save-lfp-data Enable saving of LFP data (overrides config)"
  echo "  -m, --save-metadata             Enable saving of channel selection metadata"
  echo "  -o, --overwrite, --overwrite-existing   Overwrite existing session output folders"
  echo "  -X, --cleanup, --cleanup-after  Clean up cache after processing each session"
  echo "  -d, --debug                     Enable debug mode (debugpy listening on port 5678)"
  echo ""
  echo "Examples:"
  echo "  ./run_pipeline.sh                          # Run all datasets with all stages"
  echo "  ./run_pipeline.sh subset ibl              # Run only the IBL dataset"
  echo "  ./run_pipeline.sh subset ibl,abi_visual_behaviour   # Run IBL and ABI Visual Behaviour"
  echo "  ./run_pipeline.sh debug ibl               # Debug the IBL dataset"
  echo "  ./run_pipeline.sh subset ibl -s           # Run IBL and save LFP"
  echo "  ./run_pipeline.sh subset ibl --save-lfp   # Same as above with descriptive flags"
  echo "  ./run_pipeline.sh subset ibl -fg          # Run IBL with only global event detection using existing probe events"
  echo ""
  echo "Environment Variables:"
  echo "  DATASET_TO_PROCESS         Alternative way to specify dataset to process"
  echo "  OUTPUT_DIR                 Base output directory for results"
  echo "  CONFIG_PATH                Custom path to configuration YAML file"
  echo ""
}

# Check if help is requested
if [[ "$1" == "--help" || "$1" == "-h" ]]; then
  show_help
  exit 0
fi

# Parse command-line options
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PYTHON_CMD="python"
CONFIG_FILE="united_detector_config.yaml"
CLEANUP_AFTER=false
SAVE_LFP=false
SAVE_CHANNEL_METADATA=true
OVERWRITE_EXISTING=false
DEBUG_MODE=false
FIND_GLOBAL=false

# Initialize defaults
COMMAND="all"
DATASETS="ibl,abi_visual_behaviour,abi_visual_coding"

# Handle positional arguments for dataset selection
if [[ "$1" == "all" ]]; then
  COMMAND="all"
  shift
elif [[ "$1" == "subset" && -n "$2" ]]; then
  COMMAND="subset"
  DATASETS="$2"
  shift 2
elif [[ "$1" == "debug" && -n "$2" ]]; then
  COMMAND="subset"
  DATASETS="$2"
  DEBUG_MODE=true
  shift 2
fi

# Parse remaining options
while [[ $# -gt 0 ]]; do
  case "$1" in
    -c|--config)
      CONFIG_FILE="$2"
      shift 2
      ;;
    -fg|--find-global)
      FIND_GLOBAL=true
      shift
      ;;
    -s|--save-lfp|--save-lfp-data)
      SAVE_LFP=true
      shift
      ;;
    -m|--save-metadata|--save-channel-metadata)
      SAVE_CHANNEL_METADATA=true
      shift
      ;;
    -o|--overwrite|--overwrite-existing)
      OVERWRITE_EXISTING=true
      shift
      ;;
    -X|--cleanup|--cleanup-after)
      CLEANUP_AFTER=true
      shift
      ;;
    -d|--debug)
      DEBUG_MODE=true
      shift
      ;;
    -h|--help)
      show_help
      exit 0
      ;;
    --)
      shift
      break
      ;;
    *)
      break
      ;;
  esac
done

# Validate dataset selection
if [[ "$COMMAND" == "subset" && -z "$DATASETS" ]]; then
  echo "Error: 'subset' command requires a dataset specification."
  echo "Example: ./run_pipeline.sh subset ibl"
  exit 1
fi

# Decide which dataset(s) to process
if [[ "$COMMAND" == "subset" ]]; then
  # Check if there's a comma in DATASETS
  if [[ "$DATASETS" == *","* ]]; then
    # Multiple datasets specified
    export DATASET_TO_PROCESS="$DATASETS"
  else
    # Single dataset specified
    export DATASET_TO_PROCESS="$DATASETS"
  fi
else
  # Process all datasets
  export DATASET_TO_PROCESS="all"
fi

# Override with environment variable if explicitly set
if [[ -n "${DATASET_TO_PROCESS_ENV}" ]]; then
  export DATASET_TO_PROCESS="${DATASET_TO_PROCESS_ENV}"
fi

# Output configuration
echo "========================================================"
echo "Starting SWR detection pipeline with:"
echo "- Command: $COMMAND"
echo "- Datasets to process: $DATASETS"
echo "- DATASET_TO_PROCESS: $DATASET_TO_PROCESS"
echo "- Debug mode: $DEBUG_MODE"
echo "- Configuration file: $CONFIG_FILE"
if [[ "$FIND_GLOBAL" == "true" ]]; then echo "- Running global event detection using existing probe events"; fi
if [[ "$SAVE_LFP" == "true" ]]; then echo "- Saving LFP data"; fi
if [[ "$SAVE_CHANNEL_METADATA" == "true" ]]; then echo "- Saving channel selection metadata"; fi
if [[ "$OVERWRITE_EXISTING" == "true" ]]; then echo "- Overwriting existing data"; fi
if [[ "$CLEANUP_AFTER" == "true" ]]; then echo "- Cleaning up after processing"; fi
echo "========================================================"

# Create a timestamp for this run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_ID="${RUN_NAME}_${TIMESTAMP}"

# Log directory goes under $OUTPUT_DIR/logs/$RUN_ID
export LOG_DIR="${OUTPUT_DIR}/logs/${RUN_ID}"

# Detector config path defaults to a local file in the current working dir
export CONFIG_PATH=${CONFIG_PATH:-"$(pwd)/united_detector_config.yaml"}

# Build command-line arguments for the Python script
PYTHON_ARGS=""
if [[ "$FIND_GLOBAL" == "true" ]]; then PYTHON_ARGS+=" --find-global"; fi
if [[ "$SAVE_LFP" == "true" ]]; then PYTHON_ARGS+=" --save-lfp"; fi
if [[ "$SAVE_CHANNEL_METADATA" == "true" ]]; then PYTHON_ARGS+=" --save-channel-metadata"; fi
if [[ "$OVERWRITE_EXISTING" == "true" ]]; then PYTHON_ARGS+=" --overwrite-existing"; fi
if [[ "$CLEANUP_AFTER" == "true" ]]; then PYTHON_ARGS+=" --cleanup-after"; fi
if [[ "$DEBUG_MODE" == "true" ]]; then PYTHON_ARGS+=" --debug"; fi
PYTHON_ARGS+=" --config $CONFIG_PATH"

# Create output/log directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

# Activate environment helper
activate_environment() {
  local dataset="$1"
  if [[ "$dataset" == "ibl" ]]; then
    echo "Activating ONE_ibl_env environment for IBL dataset"
    if [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
      source "${HOME}/miniconda3/etc/profile.d/conda.sh"
      conda activate ONE_ibl_env
    elif [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
      source "${HOME}/anaconda3/etc/profile.d/conda.sh"
      conda activate ONE_ibl_env
    else
      source activate ONE_ibl_env || conda activate ONE_ibl_env
    fi
  elif [[ "$dataset" == "abi_visual_behaviour" || "$dataset" == "abi_visual_coding" ]]; then
    echo "Activating allensdk_env environment for Allen Brain Institute dataset"
    if [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
      source "${HOME}/miniconda3/etc/profile.d/conda.sh"
      conda activate allensdk_env
    elif [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
      source "${HOME}/anaconda3/etc/profile.d/conda.sh"
      conda activate allensdk_env
    else
      source activate allensdk_env || conda activate allensdk_env
    fi
  else
    echo "Error: Unknown dataset '$dataset'. Cannot determine environment."
    return 1
  fi

  if [ $? -ne 0 ]; then
    echo "Failed to activate conda environment for: $dataset"
    return 1
  fi
  return 0
}

# Process each dataset in turn
PIPELINE_STATUS=0
IFS=',' read -ra DATASET_ARRAY <<< "$DATASETS"

for dataset in "${DATASET_ARRAY[@]}"; do

  echo "========================================================"
  echo "Processing dataset: $dataset"
  echo "========================================================"
  
  export DATASET_TO_PROCESS="$dataset"
  export POOL_SIZE="$CORES"
  export LOG_FILE="$LOG_DIR/${dataset}.log"
  
  # Activate environment
  activate_environment "$dataset"
  if [ $? -ne 0 ]; then
    PIPELINE_STATUS=1
    continue
  fi
  
  # If debug mode, wait for you to attach
  if [ "$DEBUG_MODE" = true ]; then
    echo "Debug mode active. Starting swr_neuropixels_detector_main.py with debugpy."
    echo "Attach using VS Code's 'Attach to Running Script' on port 5678..."
    export DEBUG_MODE="true"
    sleep 2
  fi
  
  echo "Running swr_neuropixels_detector_main.py for $dataset with $CORES cores..."
  python swr_neuropixels_detector_main.py $PYTHON_ARGS 2>&1 | tee "$LOG_FILE"
  DETECTOR_STATUS=$?
  
  if [ $DETECTOR_STATUS -eq 0 ]; then
    echo "Processing of $dataset completed successfully at $(date)"
  else
    echo "Error: Processing of $dataset failed with status $DETECTOR_STATUS."
    echo "Check log file: $LOG_FILE"
    tail -n 30 "$LOG_FILE"
    PIPELINE_STATUS=1
    echo "Continuing with next dataset despite failure..."
  fi
  
  # Deactivate
  conda deactivate
done

# Final status
if [ $PIPELINE_STATUS -eq 0 ]; then
  echo "========================================================"
  echo "Pipeline completed successfully."
  echo "All datasets processed sequentially: $DATASETS"
  echo "Log files in: $LOG_DIR"
  echo "Completed at $(date)"
else
  echo "========================================================"
  echo "Pipeline failed. See logs for details: $LOG_DIR"
fi

exit $PIPELINE_STATUS