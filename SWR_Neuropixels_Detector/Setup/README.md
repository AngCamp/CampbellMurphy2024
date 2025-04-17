# Project Setup

This directory contains resources and information related to setting up the environment for the SWR Neuropixels Detector project.

## Configuration

The main configuration file `united_detector_config.yaml` (or a user-specified one) controls various aspects of the pipeline, including paths, dataset selection, processing parameters, and flags.

Ensure that the paths specified in the configuration file (e.g., `swr_output_dir`, `lfp_output_dir`, `session_list_file`, cache directories) are correct for your system.

## Environment Setup (Conda)

The pipeline relies on specific Python packages. We recommend using Conda to manage the environment.

1.  **Install Conda:** If you don't have Conda, install Miniconda or Anaconda from [https://docs.conda.io/projects/miniconda/en/latest/](https://docs.conda.io/projects/miniconda/en/latest/) or [https://www.anaconda.com/products/distribution](https://www.anaconda.com/products/distribution).

2.  **Create Environment:** Create a Conda environment using the provided `environment.yml` file (if available) or install packages manually. The specific environment name might be specified in the config file (`run_details.conda_env_name`).

    ```bash
    # Example using an environment file (if provided)
    # conda env create -f environment.yml 

    # Example creating manually (adjust package versions as needed)
    conda create --name swr_env python=3.9 numpy pandas scipy pyyaml matplotlib tqdm 
    conda activate swr_env
    # Install dataset-specific packages
    # For AllenSDK (Visual Behaviour / Visual Coding)
    pip install allensdk
    # For IBL
    pip install ONE-api ibllib iblscripts iblatlas ibldsp spikeglx
    # For Ripple Detection
    pip install ripple_detection
    ```

3.  **Activate Environment:** Before running the pipeline, activate the environment:
    ```bash
    conda activate <your_environment_name> 
    ```

## MNE Environment (Optional - For Filter Design)

If you intend to design your own bandpass filters (e.g., for the ripple band or sharp-wave component) using the MNE-Python library, you will need an environment with MNE installed. This is separate from the main pipeline execution environment unless you combine them.

1.  **Create MNE Environment:**
    ```bash
    # Create a new conda environment (e.g., named 'mne')
    conda create --name mne --channel=conda-forge --override-channels python=3.9 mne matplotlib scipy notebook ipython pandas
    ```

2.  **Activate MNE Environment:**
    ```bash
    conda activate mne
    ```

3.  **Run Jupyter Notebook:** You can then run the filter design notebook (e.g., `PowerBandFIlters/mne_filter_design_1500hz.ipynb`):
    ```bash
    jupyter notebook
    ```

Refer to the official MNE installation guide for the most up-to-date instructions: [https://mne.tools/stable/install/manual_install.html](https://mne.tools/stable/install/manual_install.html)

*(Note: Installing MNE in the main pipeline environment is also possible but might increase complexity and potential conflicts.)*

## Running the Pipeline

Use the `run_pipeline.sh` script located in the main `SWR_Neuropixels_Detector` directory. See the main project README for detailed usage instructions. 