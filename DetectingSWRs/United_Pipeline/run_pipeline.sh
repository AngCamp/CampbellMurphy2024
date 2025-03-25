# Bash code
# General Settings
#!/bin/bash

# Set environment variables with defaults
export DATASET_TO_PROCESS=${1:-"ibl"}
export RUN_NAME=${RUN_NAME:-"on_the_cluster"}
export OUTPUT_DIR=${OUTPUT_DIR:-"/space/scratch/SWR_final_pipeline/testing_dir"}
export ABI_VISUAL_CODING_SDK_CACHE=${ABI_VISUAL_CODING_SDK_CACHE:-"/space/scratch/allen_viscoding_data"}
export ABI_VISUAL_BEHAVIOUR_SDK_CACHE=${ABI_VISUAL_BEHAVIOUR_SDK_CACHE:-"/space/scratch/allen_visbehave_data"}
export IBL_ONEAPI_CACHE=${IBL_ONEAPI_CACHE:-"/space/scratch/IBL_data_cache"}

# Run your code with the environment variables set
#python your_script.py  # or snakemake command