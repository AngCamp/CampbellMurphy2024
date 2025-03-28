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
  echo "  --help    Show this help message"
  echo ""
  echo "Parameters:"
  echo "  datasets  Required with 'subset' command. Comma-separated list of datasets."
  echo "            Examples: 'ibl' or 'abi_visual_behaviour,abi_visual_coding'"
  echo ""
  echo "Options:"
  echo "  --keep-smk-report    Keep Snakemake report files in the output directory"
  echo "  --cores N        Number of cores for Snakemake to use (default: 8)"
  echo ""
  echo "Examples:"
  echo "  ./run_pipeline.sh                          # Run all datasets"
  echo "  ./run_pipeline.sh all                      # Same as above"
  echo "  ./run_pipeline.sh all --keep-smk-report        # Run all datasets and keep report files"
  echo "  ./run_pipeline.sh subset ibl               # Run only IBL dataset"
  echo "  ./run_pipeline.sh subset ibl --keep-smk-report # Run only IBL dataset and keep report"
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

# Initialize default values
COMMAND="all"
DATASETS="ibl,abi_visual_behaviour,abi_visual_coding"
CORES=8
KEEP_REPORT=false

# Parse command line arguments
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
    --keep-smk-report)
      KEEP_REPORT=true
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
      # Check if it might be a number intended for --cores
      if [[ "$arg" =~ ^[0-9]+$ ]]; then
        CORES="$arg"
      else
        echo "Warning: Unknown argument '$arg'. Ignoring."
      fi
      ;;
  esac
  
  i=$((i+1))
done

# Check if we're using subset command without specifying datasets
if [ "$COMMAND" = "subset" ] && [ -z "$DATASETS" ]; then
  echo "Error: 'subset' command requires a list of datasets."
  exit 1
fi

# Set environment type based on datasets
if [ "$COMMAND" = "subset" ]; then
  if [[ "$DATASETS" == *","* ]]; then
    export DATASET_TO_PROCESS="all"
  else
    export DATASET_TO_PROCESS="$DATASETS"
  fi
else
  export DATASET_TO_PROCESS="all"
fi

# Set default environment variables
export RUN_NAME=${RUN_NAME:-"on_the_cluster"}

# Create a timestamp for this run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_ID="${RUN_NAME}_${TIMESTAMP}"

# Get parent directory of OUTPUT_DIR for placing temp_wd
export OUTPUT_DIR=${OUTPUT_DIR:-"/space/scratch/SWR_final_pipeline/testing_dir"}
OUTPUT_PARENT=$(dirname "$OUTPUT_DIR")
export TEMP_WD="${OUTPUT_PARENT}/temp_wd_${RUN_ID}"
export LOG_DIR="${OUTPUT_DIR}/logs/${RUN_ID}"
export SNAKEMAKE_OUT="${TEMP_WD}/snakemake_output"

# Set SDK/API cache locations
export ABI_VISUAL_CODING_SDK_CACHE=${ABI_VISUAL_CODING_SDK_CACHE:-"/space/scratch/allen_viscoding_data"}
export ABI_VISUAL_BEHAVIOUR_SDK_CACHE=${ABI_VISUAL_BEHAVIOUR_SDK_CACHE:-"/space/scratch/allen_visbehave_data"}
export IBL_ONEAPI_CACHE=${IBL_ONEAPI_CACHE:-"/space/scratch/IBL_data_cache"}

# Create directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"
mkdir -p "$TEMP_WD"
mkdir -p "$TEMP_WD/tmp"
mkdir -p "$SNAKEMAKE_OUT"

# Export temp directory variables for Python and libraries
export TMPDIR="${TEMP_WD}/tmp"
export TEMP="${TEMP_WD}/tmp"
export TMP="${TEMP_WD}/tmp"

# Get the absolute path to the script directory
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")

echo "========================================================"
echo "Starting SWR detection pipeline with:"
echo "- Command: $COMMAND"
echo "- Environment type: $DATASET_TO_PROCESS"
echo "- Datasets to process: $DATASETS"
echo "- Cores: $CORES"
echo "- Output directory: $OUTPUT_DIR"
echo "- Log directory: $LOG_DIR"
echo "- Temporary working directory: $TEMP_WD"
echo "- Keep Snakemake report: $KEEP_REPORT"
echo "========================================================"

# Improve the copying of pipeline files to temp directory
echo "Copying pipeline files to temporary directory..."

# Use rsync if available, otherwise use cp with special handling
if command -v rsync &> /dev/null; then
    rsync -a "$SCRIPT_DIR/" "$TEMP_WD/" --exclude ".*" --exclude "__pycache__"
else
    # First copy all visible files and directories
    cp -r "$SCRIPT_DIR/"* "$TEMP_WD/" 2>/dev/null || true
    
    # Copy specific important directories that might be hidden
    for dir in session_id_lists Filters; do
        if [ -d "$SCRIPT_DIR/$dir" ]; then
            mkdir -p "$TEMP_WD/$dir"
            cp -r "$SCRIPT_DIR/$dir/"* "$TEMP_WD/$dir/" 2>/dev/null || true
        fi
    done
fi

# Verify the important files are copied
echo "Verifying essential files in temporary directory:"
if [ -f "$TEMP_WD/united_swr_detector.py" ] && [ -f "$TEMP_WD/snakefile.smk" ]; then
    echo "✓ Essential pipeline files found"
else
    echo "× Warning: Some essential files may be missing!"
    if [ ! -f "$TEMP_WD/united_swr_detector.py" ]; then
        echo "× Missing: united_swr_detector.py"
        # Try to copy it directly
        cp "$SCRIPT_DIR/united_swr_detector.py" "$TEMP_WD/" 2>/dev/null || true
    fi
    if [ ! -f "$TEMP_WD/snakefile.smk" ]; then
        echo "× Missing: snakefile.smk"
        # Try to copy it directly
        cp "$SCRIPT_DIR/snakefile.smk" "$TEMP_WD/" 2>/dev/null || true
    fi
fi

# Copy loader files explicitly
for loader in IBL_loader.py ABI_visual_behaviour_loader.py ABI_visual_coding_loader.py; do
    if [ -f "$SCRIPT_DIR/$loader" ]; then
        cp "$SCRIPT_DIR/$loader" "$TEMP_WD/" 2>/dev/null || true
        echo "✓ Copied $loader"
    fi
done

# List key files for verification
echo "Key files in temporary directory:"
ls -la "$TEMP_WD/"*.py "$TEMP_WD/"*.smk "$TEMP_WD/"*.yaml 2>/dev/null || true

# Modify the Snakefile to use SNAKEMAKE_OUT directory for results
sed -i "s|\"results/|\"${SNAKEMAKE_OUT}/|g" "$TEMP_WD/snakefile.smk" 2>/dev/null || true

# Change to temp working directory for the pipeline run
cd "$TEMP_WD"

# Run the pipeline from the temp directory
snakemake -s snakefile.smk --configfile united_detector_config.yaml \
  --config datasets="$DATASETS" \
  --cores "$CORES" \
  --use-conda

# Store the exit status
PIPELINE_STATUS=$?

# Generate the report only if pipeline succeeded
if [ $PIPELINE_STATUS -eq 0 ]; then
    echo "Pipeline completed successfully. Generating report..."
    
    # Create an empty report file first if it doesn't exist
    if [ ! -f "final_report.html" ]; then
        echo "<html><body><h1>SWR Pipeline Report</h1></body></html>" > final_report.html
    fi
    
    snakemake -s snakefile.smk --configfile united_detector_config.yaml \
        final_report.html \
        --cores 1
    
    echo "========================================================"
    echo "Processing completed successfully!"
else
    echo "========================================================"
    echo "Pipeline failed with exit code $PIPELINE_STATUS"
    echo "Check logs for details: $LOG_DIR"
fi

# Return to original directory
cd "$SCRIPT_DIR"

# Always remove the temporary working directory
if [ $PIPELINE_STATUS -eq 0 ]; then
    echo "Cleaning up temporary working directory..."
    rm -rf "$TEMP_WD"
else
    echo "Keeping temporary files for debugging in: $TEMP_WD"
fi