# Project Setup

This directory contains resources and information related to setting up the environment for the SWR Neuropixels Detector project, this will allow you to run the notebooks in Data Usage, or if you chose to modify the pipeline and detect your own set of high frequency events.

## Environment Management

We use **Conda** as our environment manager for this project. While we chose Conda because most researchers are familiar with it, please note that we primarily use `pip` for package installations within the conda environments as most do not have a Conda installation available.

**Important:** The pipeline expects specific environment names (`allensdk_env` and `ONE_ibl_env`) as these are hardcoded in the `run_pipeline.sh` script. Please use the exact names provided by our setup script.

## Prerequisites

**Install Conda:** If you don't have Conda installed, please install Miniconda or Anaconda:
- Miniconda (recommended): [https://docs.conda.io/projects/miniconda/en/latest/](https://docs.conda.io/projects/miniconda/en/latest/)
- Anaconda: [https://www.anaconda.com/products/distribution](https://www.anaconda.com/products/distribution)

## Automated Environment Setup

We provide a single setup script that creates both required environments. Simply run:

```bash
cd Setup/
./setup_environments.sh
```

If you get a permission error, make the script executable first:
```bash
chmod +x setup_environments.sh
./setup_environments.sh
```

### What the Setup Script Does

The `setup_environments.sh` script performs the following steps:

1. **Checks for Conda installation** and exits with helpful error message if not found
2. **Makes itself executable** (includes `chmod +x` command)
3. **Creates `allensdk_env` environment** with Python 3.10 for Allen Institute datasets:
   - Installs AllenSDK==2.16.2 via pip (brings numpy, pandas, scipy, matplotlib, tqdm as dependencies)
   - Installs pyyaml=6.0.2 via conda (pipeline requirement)
   - Installs ripple_detection=1.5.1 from conda-forge (edeno channel)
   - Sets up Jupyter kernel support with ipykernel==6.26.0
4. **Creates `ONE_ibl_env` environment** with Python 3.10 for IBL datasets:
   - Installs ONE-api==3.0.0 via pip (brings numpy, pandas, pyyaml, tqdm as dependencies)
   - Installs ibllib==3.3.0 via pip (brings matplotlib, seaborn as dependencies)
   - Installs ibl-neuropixel==1.8.1 via pip
   - Installs ripple_detection=1.5.1 from conda-forge (edeno channel)
   - Sets up Jupyter kernel support with ipykernel==6.26.0
5. **Configures Jupyter kernels** so you can select the appropriate environment in notebooks

### Package Dependencies

The setup script uses a minimal approach - it only installs the essential packages and lets the main packages handle their own dependencies:

**allensdk_env core packages come from:**
- **numpy, pandas, scipy, matplotlib, seaborn, tqdm** → installed as AllenSDK dependencies
- **pyyaml** → installed separately for pipeline requirements
- **ripple_detection** → specialized package from conda-forge
- **ipykernel** → for Jupyter notebook support

**ONE_ibl_env core packages come from:**
- **numpy, pandas, pyyaml, tqdm** → installed as ONE-api dependencies  
- **matplotlib, seaborn** → installed as ibllib dependencies
- **ripple_detection** → specialized package from conda-forge
- **ipykernel** → for Jupyter notebook support

This approach ensures compatibility and reduces installation conflicts by letting each main package manage its own dependency versions.

### Using the Environments

After setup, you can activate the environments manually:

```bash
# For Allen Institute datasets
conda activate allensdk_env

# For IBL datasets  
conda activate ONE_ibl_env
```

**Note:** The `run_pipeline.sh` script automatically activates the correct environment based on the dataset being processed, so manual activation is only needed for interactive work or running individual notebooks.

### Jupyter Notebook Support

Both environments are configured as Jupyter kernels:
- **Python (allensdk_env)** - for Allen Institute data analysis
- **Python (ONE_ibl_env)** - for IBL data analysis

Select the appropriate kernel when working with notebooks.

## Configuration

The main configuration file `united_detector_config.yaml` (or a user-specified one) controls various aspects of the pipeline, including paths, dataset selection, processing parameters, and flags.

Ensure that the paths specified in the configuration file (e.g., `swr_output_dir`, `lfp_output_dir`, `session_list_file`, cache directories) are correct for your system.

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