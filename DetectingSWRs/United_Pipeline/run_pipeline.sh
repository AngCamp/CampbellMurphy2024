#!/bin/bash

# Set environment variables with defaults
export DATASET_TO_PROCESS=${1:-"all"}
export RUN_NAME=${RUN_NAME:-"on_the_cluster"}
export OUTPUT_DIR=${OUTPUT_DIR:-"/space/scratch/SWR_final_pipeline/testing_dir"}
export ABI_VISUAL_CODING_SDK_CACHE=${ABI_VISUAL_CODING_SDK_CACHE:-"/space/scratch/allen_viscoding_data"}
export ABI_VISUAL_BEHAVIOUR_SDK_CACHE=${ABI_VISUAL_BEHAVIOUR_SDK_CACHE:-"/space/scratch/allen_visbehave_data"}
export IBL_ONEAPI_CACHE=${IBL_ONEAPI_CACHE:-"/space/scratch/IBL_data_cache"}

# Get dataset parameter and number of cores
DATASETS=${2:-"ibl,abi_visual_behaviour,abi_visual_coding"}
CORES=${3:-8}  # Default to 8 cores, adjust as needed for your system

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Run the pipeline
snakemake -s snakefile.smk --configfile united_detector_config.yaml \
  --config datasets="$DATASETS" \
  --cores "$CORES" \
  --use-conda

# Generate the report
snakemake -s snakefile.smk --configfile united_detector_config.yaml \
  final_report.html \
  --cores "$CORES"