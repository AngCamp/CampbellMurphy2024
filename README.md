# Mouse Hippocampal Sharp Wave Ripple Dataset Curated From Public Neuropixels Datasets

![](Images/figure_one_revised.png)
*Overview figure showing the dataset structure and probe placements across the three datasets (ABI Visual Behavior, ABI Visual Coding, and IBL), with a pictoral explanation of the detector pipeline showing how the values in the Events CSV relates to source data of the anatomy and electrophysiology.*

![](Images/SupplementalSWRDetectorWorkflow.png)
*Detailed workflow diagram showing the SWR detection pipeline steps and data flow, including preprocessing, detection, filtering, and output generation stages.*

## Description
A repo showcasing how to process and analyze Neuropixels LFP from the two largest publicly available datasets: the ABI Visual Behaviour and the IBL dataset.  Currently both the IBL and ABI datasets are missing NWB files on DANDI archive.  Once they are done I'd like to switch to a single pipeline that processes all of them but right now we have to use the individual APIs to get the full datasets.

The data set is available at the [OSF](https://osf.io/9gm6x/).

    ```
    wget -O "swr_dataset.zip" 
    "https://files.osf.io/v1/resources/9gm6x/providers/osfstorage/?zip="
    ```

### Data_Usage

Contains tutorials for understanding and analyzing Sharp Wave Ripple (SWR) data at different scales:

- **`swrs_allen_visual_behaviour.ipynb`**: Demonstrates single-session data alignment and analysis, showing how to load and organize SWR data for individual recording sessions. Useful for understanding data structure and performing preliminary analyses.

- **`choosing_event_thresholds.ipynb`**: Establishes methods for threshold testing and provides infrastructure for dataset-level analyses. Includes the `SharpWaveComponentPlotter` object for managing data across multiple sessions and datasets, enabling cross-dataset comparisons and formal hypothesis testing.

### SWR_Neuropixels_Detector

Contains the main pipeline for running the detection scripts. It includes a config file (`united_detector_config.yaml`) which sets input and output paths, parameters for detection (e.g., ripple envelope threshold), and filtering options. If one wishes to rerun the detection pipelines, this config file can be modified accordingly. There is also a `run_pipeline.sh` script to execute the different stages of the pipeline.

#### Example Usage
1. Start a `tmux` session (as the code can take a while to run):
    ```bash
    tmux
    ```

2. Activate the appropriate conda environment (e.g., `allensdk_env` or `ONE_ibl_env` depending on the data source targeted in the config):
    ```bash
    conda activate allensdk_env 
    # or conda activate ONE_ibl_env
    ```

3. Change directory to the pipeline folder:
    ```bash
    cd SWR_Neuropixels_Detector
    ```

4. Ensure the config file, `united_detector_config.yaml`, is set correctly, including the number of cores (`pool_size`) your machine can handle.

5. Run the pipeline script. Use flags to specify which stages to run (this will be implemented in a later step):
    ```bash
    ./run_pipeline.sh # Add flags like -p -f -g as needed later
    ```

6.  Use ctrl+b, d to exit the tmux session without killing it. It is recomended to check htop to ensure the server is behaving appropriately.

Note:  We have also created scripts for running the pipelines on slurm for shared computing clusters.  (Will be provided)

#### SWR session folder structure

The code outputs the following set of files for each session.
![](Images/figure_3_SWR_Dataset_v3.png)
*Schematic showing the output file structure for each SWR session, including event files, metadata, and channel selection information organized by probe and session.*

#### Filters

Contains the filters and a readme displaying the code used to create the filters for the SWR detection pipelines.  Uses environment for [mne package with core dependencies](https://mne.tools/stable/install/manual_install.html#installing-mne-python-with-core-dependencies).

### Figures_and_Technical_Validation

Contains scripts and workflows for generating publication figures and technical validation analyses. The subfolders contain automated pipelines that generate multiple visualizations for selection:

- **`probe_selection_validation/`**: Contains scripts for analyzing channel selection metadata and generating depth-dependent plots showing ripple band power, skewness, and sharp wave features across probe depths. Includes automated workflows for creating bar plots and selection visualizations.

- **`Sharp_wave_component_validation/`**: Contains the `SWRExplorer.py` tool and workflows for finding and visualizing the best SWR events across datasets. Includes scripts for generating multiple event visualizations and selecting the most representative examples for publication.

- **`Relating_SWR_to_other_data/`**: Contains scripts for relating SWR events to spiking activity, pupil data, and running behavior. Includes automated workflows for generating multiple raster plots and behavioral correlation analyses.

- **`Sankey_plots/`**: Contains code to generate Sankey diagrams visualizing the filtering pipeline for SWR events, showing how events flow through different filters and classifications.

- **`Mouse_subject_details/`**: Contains scripts for generating subject information summary tables (Tables 1, 2, and 3) by querying dataset APIs and SWR pipeline outputs, then creating formatted summary tables for publication.



### Images

Images for the repo.


### Setup

Contains conda environment files and setup scripts for the different datasets.



![](Images/Figure4_version2_global_event_4_session_1086410738_id_2250.png)
*Example of a global SWR event visualization showing the event across multiple probes, with LFP traces, power (z-scored), and event timing displayed for comprehensive analysis.*

![](Images/Figure_5_MI_selection.png)
*Visualization of net ripple power based channel selection for pyramidal layer identification for ripple detection and modulation index-based channel selection for mid striatum radiatum layer identification for sharp wave detection, showing how different channels are evaluated and selected based on their modulation index values.*

![](Images/v2_real_data_filtering_sankey.png)
*Sankey diagram showing the filtering pipeline and event classification flow, illustrating how SWR events are processed through various thresholds and categorized based on their properties.*
