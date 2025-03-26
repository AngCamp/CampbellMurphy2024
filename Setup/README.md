# SWR Detection Pipeline Environment Setup

This document describes how to set up the required environments for running the SWR detection pipeline.

## Prerequisites

- Git (to clone the repository)
- Conda (Miniconda or Anaconda)

## Installing Conda

If you don't have Conda installed, follow these steps:

1. Download Miniconda:

   - **Linux**:
     ```bash
     wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
     ```
   
   - **MacOS**:
     ```bash
     wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh -O miniconda.sh
     ```

2. Install Miniconda:
   ```bash
   bash miniconda.sh -b -p $HOME/miniconda
   ```

3. Initialize conda:
   ```bash
   source $HOME/miniconda/bin/activate
   conda init
   ```

4. Restart your shell or terminal

## Installing Snakemake

Install Snakemake in your base conda environment:

```bash
conda install -c bioconda -c conda-forge snakemake
```

## Setting Up Required Environments

The SWR detection pipeline requires two different conda environments:

1. `allensdk_env`: For processing Allen Brain Institute datasets
2. `ONE_ibl_env`: For processing IBL datasets

### Creating the Environments

Navigate to the repository's root directory and create the environments:

```bash
# Create the Allen SDK environment
conda env create -f envs/allensdk_env.yml

# Create the IBL environment
conda env create -f envs/ONE_ibl_env.yml
```

### Verifying the Environments

Verify that the environments were created correctly:

```bash
conda env list
```

You should see both `allensdk_env` and `ONE_ibl_env` in the list.

## Note on Working Directory

All pipeline code is located in the `DetectionSWRs` folder. This folder should be used as the working directory when running the pipeline.

## Environment Files

The environment YAML files contain all necessary dependencies:

- `allensdk_env.yml`: Contains dependencies for Allen Brain Institute data processing
- `ONE_ibl_env.yml`: Contains dependencies for IBL data processing

These environments include all required packages, with specific versions to ensure reproducibility.