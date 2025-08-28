#!/bin/bash

# Combined setup script for SWR Neuropixels Detector environments
# This script creates both conda environments needed for the pipeline:
# - allensdk_env: For Allen Institute datasets
# - ONE_ibl_env: For International Brain Laboratory (IBL) datasets

set -e  # Exit on any error

echo "=============================================="
echo "SWR Neuropixels Detector Environment Setup"
echo "=============================================="
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Error: conda is not installed or not in PATH"
    echo "Please install Miniconda or Anaconda first:"
    echo "  https://docs.conda.io/projects/miniconda/en/latest/"
    echo "  https://www.anaconda.com/products/distribution"
    exit 1
fi

echo "✅ conda found: $(conda --version)"
echo ""

# Make this script executable (in case it wasn't already)
chmod +x "$0"

# Initialize conda for bash (needed for activation to work in script)
source $(conda info --base)/etc/profile.d/conda.sh

echo "Setting up environment 1/2: allensdk_env"
echo "=========================================="

# Create the AllenSDK environment
echo "Creating conda environment 'allensdk_env'..."
conda create --name allensdk_env python=3.10 -y

echo "Activating allensdk_env..."
conda activate allensdk_env

echo "Installing core packages via conda..."
conda install numpy pandas scipy matplotlib tqdm pyyaml -y

echo "Installing AllenSDK..."
pip install allensdk

echo "Installing ripple detection package..."
conda install -c edeno ripple_detection -y

echo "Installing Jupyter kernel support..."
pip install ipykernel
python -m ipykernel install --user --name=allensdk_env --display-name="Python (allensdk_env)"

echo "✅ allensdk_env setup complete!"
echo ""

echo "Setting up environment 2/2: ONE_ibl_env"
echo "========================================"

# Create the IBL/ONE environment
echo "Creating conda environment 'ONE_ibl_env'..."
conda create --name ONE_ibl_env python=3.10 -y

echo "Activating ONE_ibl_env..."
conda activate ONE_ibl_env

echo "Installing core packages via conda..."
conda install numpy pandas scipy matplotlib tqdm pyyaml -y

echo "Installing ONE-api..."
pip install ONE-api

echo "Installing ibllib..."
pip install ibllib

echo "Installing ibl-neuropixel version 1.8.1..."
pip install ibl-neuropixel==1.8.1

echo "Installing ripple detection package..."
conda install -c edeno ripple_detection -y

echo "Installing Jupyter kernel support..."
pip install ipykernel
python -m ipykernel install --user --name=ONE_ibl_env --display-name="Python (ONE_ibl_env)"

echo "✅ ONE_ibl_env setup complete!"
echo ""

# Deactivate any active environment
conda deactivate

echo "=============================================="
echo "🎉 Environment setup completed successfully!"
echo "=============================================="
echo ""
echo "Two environments have been created:"
echo "  1. allensdk_env    - For Allen Institute datasets"
echo "  2. ONE_ibl_env     - For IBL datasets"
echo ""
echo "To use these environments:"
echo "  conda activate allensdk_env    # For Allen Institute data"
echo "  conda activate ONE_ibl_env     # For IBL data"
echo ""
echo "For Jupyter notebooks, select the appropriate kernel:"
echo "  - Python (allensdk_env)"
echo "  - Python (ONE_ibl_env)"
echo ""
echo "Note: The run_pipeline.sh script will automatically activate"
echo "the correct environment based on the dataset being processed."
echo ""
echo "For IBL users: You may need to configure ONE-api with your"
echo "IBL credentials. See: https://int-brain-lab.github.io/ONE/"
