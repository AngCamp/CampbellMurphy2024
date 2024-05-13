#!/bin/bash

# Check if script is executable, if not, try to make it so
if [[ ! -x $0 ]]; then
    chmod +x $0 || {
        echo "Failed to make script executable"
        exit 1
    }
fi

# Variables for allen_swr_detector.py
pool_size=6
sdk_cache_dir='/space/scratch/allen_visbehave_data'
output_dir='/space/scratch/allen_visbehave_swr_data'
swr_output_dir='testing_dir'
run_name='final_run'
select_these_sessions=[]
only_brain_observatory_sessions=True
dont_wipe_these_sessions=[]
gamma_event_thresh=3
gamma_filter_path='/home/acampbell/NeuropixelsLFPOnRamp/PowerBandFilters/swr_detection_script_filters_1500Hz/frank2008_gamma_1500hz_bandpass_filter.npz'
theta_filter_path='/home/acampbell/NeuropixelsLFPOnRamp/PowerBandFilters/swr_detection_script_filters_1500Hz/theta_1500hz_bandpass_filter.npz'
ripple_band_threshold=2
movement_artifact_ripple_band_threshold=2
save_lfp=True

# Variables for Filtering_SWR.py
sdk_cache_dir_filter='/space/scratch/allen_visbehave_data'
input_dir_filter='/space/scratch/allen_visbehave_swr_data/testing_dir'
output_dir_filter='/space/scratch/allen_visbehave_swr_data/'
swr_output_dir_filter='testing_dir'

# Variables for Global_ripple_detector.py
input_dir_global='/space/scratch/allen_visbehave_swr_data/testing_dir_filtered'
output_dir_global='.'
global_rip_label='no_movement_no_gamma'
minimum_ripple_num=100

# Activate the conda environment
source /home/acampbell/miniconda3/etc/profile.d/conda.sh
conda activate allensdk_env
python_path=$(which python)

# Print debugging information
#echo "Environment Variables:"
#printenv
#echo "Working Directory:"
#pwd
#echo "Script Variables:"
#set

# Run the Python scripts with the variables as command-line arguments
echo "Running allen_swr_detector.py"
$python_path ./allen_swr_detector.py --pool_size $pool_size --sdk_cache_dir $sdk_cache_dir --output_dir $output_dir --swr_output_dir $swr_output_dir --run_name $run_name --select_these_sessions $select_these_sessions --only_brain_observatory_sessions $only_brain_observatory_sessions --dont_wipe_these_sessions $dont_wipe_these_sessions --gamma_event_thresh $gamma_event_thresh --gamma_filter_path $gamma_filter_path --theta_filter_path $theta_filter_path --ripple_band_threshold $ripple_band_threshold --movement_artifact_ripple_band_threshold $movement_artifact_ripple_band_threshold --save_lfp $save_lfp
echo "Exit status of allen_swr_detector.py: $?"
echo "Running Filtering_SWR_Events_Karlsson_detector.py"
$python_path ./Filtering_SWR_Events_Karlsson_detector.py --sdk_cache_dir_filter $sdk_cache_dir_filter --input_dir_filter $input_dir_filter --output_dir_filter $output_dir_filter --swr_output_dir_filter $swr_output_dir_filter
echo "Running Global_ripple_detector.py"
$python_path ./Global_ripple_detector.py --input_dir_global $input_dir_global --global_rip_label $global_rip_label --minimum_ripple_num $minimum_ripple_num