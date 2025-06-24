#!/bin/bash

# Figure 9 Complete Workflow Script
# This script runs all the data gathering, plotting, and caption generation steps

set -e  # Exit on any error

# Activate environment helper (same as run_pipeline.sh)
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

# Parse command line arguments
MAX_SESSIONS=""
SKIP_GATHERING=false
N_PROCESSES="--n_processes 4"  # Default to 4 processes
while [[ $# -gt 0 ]]; do
    case $1 in
        --max_sessions)
            MAX_SESSIONS="--max_sessions $2"
            shift 2
            ;;
        --n_processes)
            N_PROCESSES="--n_processes $2"
            shift 2
            ;;
        --skip-gathering)
            SKIP_GATHERING=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [--max_sessions N] [--n_processes N] [--skip-gathering]"
            echo ""
            echo "Options:"
            echo "  --max_sessions N    Limit processing to first N sessions per dataset (for debugging)"
            echo "  --n_processes N     Number of parallel processes for data gathering (default: 4)"
            echo "  --skip-gathering    Skip data gathering and go straight to plotting (for debugging)"
            echo "  --help, -h          Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                    # Process all sessions with 4 processes"
            echo "  $0 --max_sessions 1   # Process only first session per dataset (fast debugging)"
            echo "  $0 --n_processes 8    # Use 8 parallel processes for faster processing"
            echo "  $0 --skip-gathering   # Skip gathering, just plot existing data"
            echo "  $0 --max_sessions 5 --n_processes 2  # Process first 5 sessions with 2 processes"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo "=== Figure 9 Workflow Starting ==="
if [[ -n "$MAX_SESSIONS" ]]; then
    echo "Debug mode: $MAX_SESSIONS sessions per dataset"
fi
if [[ -n "$N_PROCESSES" ]]; then
    echo "Multiprocessing: $N_PROCESSES"
fi
if [[ "$SKIP_GATHERING" == true ]]; then
    echo "Skipping data gathering - using existing data files"
fi
echo ""

# Data output directory
DATA_DIR="/home/acampbell/NeuropixelsLFPOnRamp/Figures_Tables_and_Technical_Validation/Distribution_v2/distributions_for_plotting"

# Remove old data files to ensure fresh data (only if not skipping gathering)
if [[ "$SKIP_GATHERING" == false ]]; then
    echo "Removing old data files to ensure fresh data..."
    rm -f "$DATA_DIR"/abi_visbehave_*.npz
    rm -f "$DATA_DIR"/abi_viscoding_*.npz
    rm -f "$DATA_DIR"/ibl_*.npz
    echo "✓ Old data files removed"
    echo ""
fi

# Step 1: ABI Visual Behaviour Data Gathering
if [[ "$SKIP_GATHERING" == false ]]; then
    echo "Step 1: Gathering ABI Visual Behaviour data..."
    activate_environment "abi_visual_behaviour"
    python gather_validation_distributions_data.py --dataset abi_visbehave $MAX_SESSIONS $N_PROCESSES
    echo "✓ ABI Visual Behaviour data gathered"
    conda deactivate
else
    echo "Step 1: Skipping ABI Visual Behaviour data gathering"
fi

# Step 2: ABI Visual Coding Data Gathering  
if [[ "$SKIP_GATHERING" == false ]]; then
    echo "Step 2: Gathering ABI Visual Coding data..."
    activate_environment "abi_visual_coding"
    python gather_validation_distributions_data.py --dataset abi_viscoding $MAX_SESSIONS $N_PROCESSES
    echo "✓ ABI Visual Coding data gathered"
    conda deactivate
else
    echo "Step 2: Skipping ABI Visual Coding data gathering"
fi

# Step 3: IBL Data Gathering
if [[ "$SKIP_GATHERING" == false ]]; then
    echo "Step 3: Gathering IBL data..."
    activate_environment "ibl"
    python gather_validation_distributions_data.py --dataset ibl $MAX_SESSIONS $N_PROCESSES
    echo "✓ IBL data gathered"
    conda deactivate
else
    echo "Step 3: Skipping IBL data gathering"
fi

# Step 4: Plotting and Caption Generation
echo "Step 4: Generating plots, extracting KS test results, and generating caption..."
activate_environment "abi_visual_behaviour"  # Use allensdk_env for plotting
python plot_figure9.py
echo "✓ Plots generated, KS results extracted, and caption created"
conda deactivate

echo "=== Figure 9 Workflow Complete ==="
echo ""
echo "Output files:"
echo "- figure9_combined_*.png and figure9_combined_*.svg (the combined figure)"
echo "- individual_subplots/ (individual subplot files)"
echo "- figure9_ks_results.json (KS test results for all distributions)"
echo "- figure9_caption.txt (caption with actual values)"
echo ""
echo "All files are saved in: $(pwd)" 