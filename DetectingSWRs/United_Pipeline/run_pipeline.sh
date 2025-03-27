#!/bin/bash

# Display help information
show_help() {
  echo "SWR Detection Pipeline"
  echo "======================================================"
  echo "Usage: ./run_pipeline.sh [command] [datasets] [cores]"
  echo ""
  echo "Commands:"
  echo "  all       Process all datasets (default if no command provided)"
  echo "  subset    Process only specified datasets (requires datasets parameter)"
  echo "  --help    Show this help message"
  echo ""
  echo "Parameters:"
  echo "  datasets  Required with 'subset' command. Comma-separated list of datasets."
  echo "            Examples: 'ibl' or 'abi_visual_behaviour,abi_visual_coding'"
  echo ""
  echo "  cores     Internal parameter for Snakemake (default: 8)"
  echo "            Note: This is a required technical parameter for Snakemake"
  echo "            but actual parallelism is controlled by pool_sizes in the config"
  echo ""
  echo "Examples:"
  echo "  ./run_pipeline.sh                          # Run all datasets"
  echo "  ./run_pipeline.sh all                      # Same as above"
  echo "  ./run_pipeline.sh subset ibl               # Run only IBL dataset"
  echo "  ./run_pipeline.sh subset \"abi_visual_behaviour,abi_visual_coding\"  # Run only ABI datasets"
  echo ""
  echo "Environment Variables:"
  echo "  RUN_NAME                      Name of the run (default: on_the_cluster)"
  echo "  OUTPUT_DIR                    Output directory for results"
  echo "  LOG_DIR                       Directory for log files (default: OUTPUT_DIR/logs)"
  echo "  ABI_VISUAL_CODING_SDK_CACHE   Cache directory for Allen Visual Coding data"
  echo "  ABI_VISUAL_BEHAVIOUR_SDK_CACHE  Cache directory for Allen Visual Behavior data"
  echo "  IBL_ONEAPI_CACHE              Cache directory for IBL data"
  echo ""
}

# Check for help flag
if [[ "$1" == "--help" || "$1" == "-h" ]]; then
  show_help
  exit 0
fi

# Default command is "all"
COMMAND=${1:-"all"}

# Set datasets based on command
if [[ "$COMMAND" == "all" ]]; then
  # Process all datasets
  DATASETS="ibl,abi_visual_behaviour,abi_visual_coding"
  CORES=${2:-8}  # If command is "all", second parameter is cores
  # Set to "all" for environment variable
  export DATASET_TO_PROCESS="all"
elif [[ "$COMMAND" == "subset" ]]; then
  # Check if datasets parameter is provided
  if [[ -z "$2" ]]; then
    echo "Error: 'subset' command requires a list of datasets."
    echo "Example: ./run_pipeline.sh subset \"ibl,abi_visual_behaviour\""
    exit 1
  fi
  
  DATASETS=$2
  CORES=${3:-8}  # If command is "subset", third parameter is cores
  
  # Set environment type based on datasets
  # If only one dataset, use that as env type, otherwise use "all"
  if [[ "$DATASETS" == *","* ]]; then
    export DATASET_TO_PROCESS="all"
  else
    export DATASET_TO_PROCESS=$DATASETS
  fi
else
  echo "Error: Unknown command '$COMMAND'"
  echo "Use 'all', 'subset', or '--help' for help"
  exit 1
fi

# Set other environment variables
export RUN_NAME=${RUN_NAME:-"on_the_cluster"}
export OUTPUT_DIR=${OUTPUT_DIR:-"/space/scratch/SWR_final_pipeline/testing_dir"}
export LOG_DIR=${LOG_DIR:-"${OUTPUT_DIR}/logs"}
export ABI_VISUAL_CODING_SDK_CACHE=${ABI_VISUAL_CODING_SDK_CACHE:-"/space/scratch/allen_viscoding_data"}
export ABI_VISUAL_BEHAVIOUR_SDK_CACHE=${ABI_VISUAL_BEHAVIOUR_SDK_CACHE:-"/space/scratch/allen_visbehave_data"}
export IBL_ONEAPI_CACHE=${IBL_ONEAPI_CACHE:-"/space/scratch/IBL_data_cache"}

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

echo "========================================================"
echo "Starting SWR detection pipeline with:"
echo "- Command: $COMMAND"
echo "- Environment type: $DATASET_TO_PROCESS"
echo "- Datasets to process: $DATASETS"
# Internal note: Cores parameter is only needed for Snakemake's internal mechanisms
# but doesn't affect our actual parallelism which is controlled by pool_sizes
echo "- Output directory: $OUTPUT_DIR"
echo "- Log directory: $LOG_DIR"
echo "========================================================"

# Run the pipeline
snakemake -s snakefile.smk --configfile united_detector_config.yaml \
  --config datasets="$DATASETS" \
  --cores "$CORES" \
  --use-conda

# Store the exit status
PIPELINE_STATUS=$?

# Generate the report only if pipeline succeeded
if [ $PIPELINE_STATUS -eq 0 ]; then
  echo "Pipeline completed successfully. Generating report..."
  snakemake -s snakefile.smk --configfile united_detector_config.yaml \
    final_report.html \
    --cores 1
  
  echo "Report generated: final_report.html"
  echo "========================================================"
  echo "Processing completed successfully!"
else
  echo "========================================================"
  echo "Pipeline failed with exit code $PIPELINE_STATUS"
  echo "Check logs for details: $LOG_DIR"
fi