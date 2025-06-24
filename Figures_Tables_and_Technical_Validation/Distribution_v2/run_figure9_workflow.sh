#!/bin/bash

# Figure 9 Complete Workflow Script
# This script runs all the data gathering, plotting, and caption generation steps

set -e  # Exit on any error

echo "=== Figure 9 Workflow Starting ==="

# Step 1: ABI Visual Behaviour Data Gathering
echo "Step 1: Gathering ABI Visual Behaviour data..."
conda activate allensdk_env
python gather_speed_theta_data_abi_visbehave.py
echo "✓ ABI Visual Behaviour data gathered"

# Step 2: ABI Visual Coding Data Gathering  
echo "Step 2: Gathering ABI Visual Coding data..."
# Note: Still in allensdk_env but using different loader/cache
python gather_speed_theta_data_abi_viscoding.py
echo "✓ ABI Visual Coding data gathered"

# Step 3: IBL Data Gathering
echo "Step 3: Gathering IBL data..."
conda activate ONE_ibl_env
python gather_speed_theta_data_ibl.py
echo "✓ IBL data gathered"

# Step 4: Plotting (can run in any environment with required packages)
echo "Step 4: Generating plots and extracting KS test results..."
conda activate allensdk_env  # or any environment with matplotlib, fitter, etc.
python plot_figure9.py
echo "✓ Plots generated and KS results extracted"

# Step 5: Caption Generation
echo "Step 5: Generating figure caption with statistical results..."
python generate_caption.py
echo "✓ Caption generated"

echo "=== Figure 9 Workflow Complete ==="
echo ""
echo "Output files:"
echo "- figure9.png and figure9.svg (the figure)"
echo "- figure9_ks_results.json (KS test results)"
echo "- figure9_caption_with_results.txt (caption with actual values)"
echo ""
echo "All files are saved in: $(pwd)" 