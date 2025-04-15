#!/bin/bash

# Display help information
show_help() {
  echo "SWR Detection Pipeline"
  echo "======================================================"
  echo "Usage: ./run_pipeline.sh [command] [datasets] [options]"
  echo ""
  echo "Commands:"
  echo "  all       Process all datasets (default if no command provided)"
  echo "  subset    Process only specified datasets (requires datasets parameter)"
  echo "  debug     Run in debug mode (single core, with pause for debugger)"
  echo "  --help    Show this help message"
  echo ""
  echo "Parameters:"
  echo "  datasets  Required with 'subset' command. Comma-separated list of datasets."
  echo "            Examples: 'ibl' or 'abi_visual_behaviour,abi_visual_coding'"
  echo ""
  echo "Options:"
  echo "  --cores N        Number of cores to use (default: 6)"
  echo ""
  echo "Examples:"
  echo "  ./run_pipeline.sh                          # Run all datasets"
  echo "  ./run_pipeline.sh all                      # Same as above"
  echo "  ./run_pipeline.sh subset ibl               # Run only IBL dataset"
  echo "  ./run_pipeline.sh debug ibl                # Run IBL dataset in debug mode (single core, paused)"
  echo ""
  echo "Environment Variables:"
  echo "  RUN_NAME                      Name of the run (default: on_the_cluster)"
  echo "  OUTPUT_DIR                    Output directory for results"
  echo "  LOG_DIR                       Directory for log files (default: OUTPUT_DIR/logs)"
  echo "  CONFIG_PATH                   Detector config path (default: ./united_detector_config.yaml)"
  echo "  ABI_VISUAL_CODING_SDK_CACHE   Cache directory for Allen Visual Coding data"
  echo "  ABI_VISUAL_BEHAVIOUR_SDK_CACHE  Cache directory for Allen Visual Behavior data"
  echo "  IBL_ONEAPI_CACHE              Cache directory for IBL data"
  echo ""
}

# Initialize defaults
COMMAND="all"
DATASETS="ibl,abi_visual_behaviour,abi_visual_coding"
CORES=6
DEBUG_MODE=false

# Parse the command line
i=1
while [ $i -le $# ]; do
  arg="${!i}"
  
  case "$arg" in
    all)
      COMMAND="all"
      ;;
    subset)
      COMMAND="subset"
      i=$((i+1))
      if [ $i -le $# ]; then
        DATASETS="${!i}"
      else
        echo "Error: 'subset' command requires a list of datasets."
        echo "Example: ./run_pipeline.sh subset \"ibl,abi_visual_behaviour\""
        exit 1
      fi
      ;;
    debug)
      COMMAND="subset"
      DEBUG_MODE=true
      i=$((i+1))
      if [ $i -le $# ]; then
        DATASETS="${!i}"
      else
        echo "Error: 'debug' command requires a dataset."
        echo "Example: ./run_pipeline.sh debug ibl"
        exit 1
      fi
      CORES=1
      ;;
    --cores)
      i=$((i+1))
      if [ $i -le $# ]; then
        CORES="${!i}"
      else
        echo "Error: --cores requires a number."
        exit 1
      fi
      ;;
    --help|-h)
      show_help
      exit 0
      ;;
    *)
      # Might be a number for --cores
      if [[ "$arg" =~ ^[0-9]+$ ]]; then
        CORES="$arg"
      else
        echo "Warning: Unknown argument '$arg'. Ignoring."
      fi
      ;;
  esac
  i=$((i+1))
done

# Validate dataset
if [ "$COMMAND" = "subset" ] && [ -z "$DATASETS" ]; then
  echo "Error: 'subset' command requires a list of datasets."
  exit 1
fi

# Decide which dataset(s) to process
if [ "$COMMAND" = "subset" ]; then
  if [[ "$DATASETS" == *","* ]]; then
    export DATASET_TO_PROCESS="all"
  else
    export DATASET_TO_PROCESS="$DATASETS"
  fi
else
  export DATASET_TO_PROCESS="all"
fi

# Set run name
export RUN_NAME=${RUN_NAME:-"on_the_cluster"}

# Create a timestamp for this run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_ID="${RUN_NAME}_${TIMESTAMP}"

# Output directory (default is /space/scratch/SWR_final_pipeline/testing_dir)
export OUTPUT_DIR=${OUTPUT_DIR:-"/space/scratch/SWR_final_pipeline/muckingabout"}

# Log directory goes under $OUTPUT_DIR/logs/$RUN_ID
export LOG_DIR="${OUTPUT_DIR}/logs/${RUN_ID}"

# Detector config path defaults to a local file in the current working dir
export CONFIG_PATH=${CONFIG_PATH:-"$(pwd)/united_detector_config.yaml"}

# Where to find dataset caches
export ABI_VISUAL_CODING_SDK_CACHE=${ABI_VISUAL_CODING_SDK_CACHE:-"/space/scratch/allen_viscoding_data"}
export ABI_VISUAL_BEHAVIOUR_SDK_CACHE=${ABI_VISUAL_BEHAVIOUR_SDK_CACHE:-"/space/scratch/allen_visbehave_data"}
export IBL_ONEAPI_CACHE=${IBL_ONEAPI_CACHE:-"/space/scratch/IBL_data_cache"}

# Create output/log directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

echo "========================================================"
echo "Starting SWR detection pipeline with:"
echo "- Command: $COMMAND"
echo "- Datasets to process: $DATASETS"
echo "- Environment type: $DATASET_TO_PROCESS"
echo "- Cores: $CORES"
echo "- Debug mode: $DEBUG_MODE"
echo "- OUTPUT_DIR: $OUTPUT_DIR"
echo "- LOG_DIR: $LOG_DIR"
echo "- CONFIG_PATH: $CONFIG_PATH"
echo "========================================================"

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
    echo "Debug mode active. Starting united_swr_detector.py with debugpy."
    echo "Attach using VS Code's 'Attach to Running Script' on port 5678..."
    export DEBUG_MODE="true"
    sleep 2
  fi
  
  echo "Running united_swr_detector.py for $dataset with $CORES cores..."
  python united_swr_detector.py 2>&1 | tee "$LOG_FILE"
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
