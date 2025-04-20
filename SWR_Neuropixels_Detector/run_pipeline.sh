#!/bin/bash

# =====================================================================
# IMPORTANT CONFIGURATION VARIABLES - EDIT THESE FOR YOUR ENVIRONMENT
# =====================================================================

# Output directory for all results
export OUTPUT_DIR=${OUTPUT_DIR:-"/space/scratch/SWR_final_pipeline/osf_campbellmurphy2025_swr_data"}

# Cache directories for datasets (where raw data is stored/downloaded)
export ABI_VISUAL_CODING_SDK_CACHE=${ABI_VISUAL_CODING_SDK_CACHE:-"/space/scratch/allen_viscoding_data"}
export ABI_VISUAL_BEHAVIOUR_SDK_CACHE=${ABI_VISUAL_BEHAVIOUR_SDK_CACHE:-"/space/scratch/allen_visbehave_data"}
export IBL_ONEAPI_CACHE=${IBL_ONEAPI_CACHE:-"/space/scratch/IBL_data_cache"}

# Run name used for organizing log files
export RUN_NAME=${RUN_NAME:-"on_the_cluster"}

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
  echo "  -h, --help                 Show this help message and exit"
  echo "  -c, --config FILE          Specify a custom configuration YAML file"
  echo "                             (default: united_detector_config.yaml)"
  echo "  -p, --run-putative         Run ONLY the Putative event detection stage"
  echo "  -f, --run-filter           Run ONLY the event Filtering stage"
  echo "  -g, --run-global           Run ONLY the Global event consolidation stage"
  echo "  -s, --save-lfp             Enable saving of LFP data (overrides config)"
  echo "  -m, --save-channel-metadata  Enable saving of channel selection metadata"
  echo "  -o, --overwrite-existing   Overwrite existing session output folders"
  echo "  -X, --cleanup-after        Clean up cache after processing each session"
  echo "  -d, --debug                Enable debug mode (debugpy listening on port 5678)"
  echo ""
  echo "Examples:"
  echo "  ./run_pipeline.sh                          # Run all datasets with all stages"
  echo "  ./run_pipeline.sh subset ibl              # Run only the IBL dataset"
  echo "  ./run_pipeline.sh subset ibl,abi_visual_behaviour   # Run IBL and ABI Visual Behaviour"
  echo "  ./run_pipeline.sh debug ibl               # Debug the IBL dataset"
  echo "  ./run_pipeline.sh subset ibl -p -s        # Run IBL with only putative stage and save LFP"
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
RUN_PUTATIVE=false
RUN_FILTER=false
RUN_GLOBAL=false
CLEANUP_AFTER=false
SAVE_LFP=false
SAVE_CHANNEL_METADATA=false
OVERWRITE_EXISTING=false
DEBUG_MODE=false

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
while getopts "c:pfgsmoXdh-:" opt; do
  case ${opt} in
    c)
      CONFIG_FILE=$OPTARG
      ;;
    p)
      RUN_PUTATIVE=true
      ;;
    f)
      RUN_FILTER=true
      ;;
    g)
      RUN_GLOBAL=true
      ;;
    s)
      SAVE_LFP=true
      ;;
    m)
      SAVE_CHANNEL_METADATA=true
      ;;
    o)
      OVERWRITE_EXISTING=true
      ;;
    X)
      CLEANUP_AFTER=true
      ;;
    d)
      DEBUG_MODE=true
      ;;
    h)
      show_help
      exit 0
      ;;
    -)
      case "${OPTARG}" in
        help)
          show_help
          exit 0
          ;;
        config)
          CONFIG_FILE="${!OPTIND}"; OPTIND=$(( $OPTIND + 1 ))
          ;;
        run-putative)
          RUN_PUTATIVE=true
          ;;
        run-filter)
          RUN_FILTER=true
          ;;
        run-global)
          RUN_GLOBAL=true
          ;;
        save-lfp)
          SAVE_LFP=true
          ;;
        save-channel-metadata)
          SAVE_CHANNEL_METADATA=true
          ;;
        overwrite-existing)
          OVERWRITE_EXISTING=true
          ;;
        cleanup-after)
          CLEANUP_AFTER=true
          ;;
        debug)
          DEBUG_MODE=true
          ;;
        *)
          echo "Invalid option: --${OPTARG}" >&2
          show_help
          exit 1
          ;;
      esac
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      show_help
      exit 1
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
if [[ "$RUN_PUTATIVE" == "true" ]]; then echo "- Running putative detection stage"; fi
if [[ "$RUN_FILTER" == "true" ]]; then echo "- Running filtering stage"; fi
if [[ "$RUN_GLOBAL" == "true" ]]; then echo "- Running global event stage"; fi
if [[ "$SAVE_LFP" == "true" ]]; then echo "- Saving LFP data"; fi
if [[ "$SAVE_CHANNEL_METADATA" == "true" ]]; then echo "- Saving channel metadata"; fi
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
if [[ "$RUN_PUTATIVE" == "true" ]]; then PYTHON_ARGS+=" --run-putative"; fi
if [[ "$RUN_FILTER" == "true" ]]; then PYTHON_ARGS+=" --run-filter"; fi
if [[ "$RUN_GLOBAL" == "true" ]]; then PYTHON_ARGS+=" --run-global"; fi
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
    break
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