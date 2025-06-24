#!/bin/bash

# Figure 9 Complete Workflow Script
# This script runs all the data gathering, plotting, and caption generation steps

set -e  # Exit on any error

# Parse command line arguments
MAX_SESSIONS=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --max_sessions)
            MAX_SESSIONS="--max_sessions $2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [--max_sessions N]"
            echo ""
            echo "Options:"
            echo "  --max_sessions N    Limit processing to first N sessions per dataset (for debugging)"
            echo "  --help, -h          Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                    # Process all sessions"
            echo "  $0 --max_sessions 1   # Process only first session per dataset (fast debugging)"
            echo "  $0 --max_sessions 5   # Process first 5 sessions per dataset"
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
echo ""

# Step 1: ABI Visual Behaviour Data Gathering
echo "Step 1: Gathering ABI Visual Behaviour data..."
conda activate allensdk_env
python gather_speed_theta_data.py --dataset abi_visbehave $MAX_SESSIONS
echo "✓ ABI Visual Behaviour data gathered"

# Step 2: ABI Visual Coding Data Gathering  
echo "Step 2: Gathering ABI Visual Coding data..."
# Note: Still in allensdk_env but using different loader/cache
python gather_speed_theta_data.py --dataset abi_viscoding $MAX_SESSIONS
echo "✓ ABI Visual Coding data gathered"

# Step 3: IBL Data Gathering
echo "Step 3: Gathering IBL data..."
conda activate ONE_ibl_env
python gather_speed_theta_data.py --dataset ibl $MAX_SESSIONS
echo "✓ IBL data gathered"

# Step 4: Plotting and Caption Generation
echo "Step 4: Generating plots, extracting KS test results, and generating caption..."
conda activate allensdk_env  # or any environment with matplotlib, fitter, etc.
python plot_figure9.py
echo "✓ Plots generated, KS results extracted, and caption created"

echo "=== Figure 9 Workflow Complete ==="
echo ""
echo "Output files:"
echo "- figure9.png and figure9.svg (the figure)"
echo "- figure9_ks_results.json (KS test results)"
echo "- figure9_caption_with_results.txt (caption with actual values)"
echo ""
echo "All files are saved in: $(pwd)" 