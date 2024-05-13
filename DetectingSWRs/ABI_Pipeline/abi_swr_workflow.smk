# Load parameters from config file
configfile: "config.yaml"

# This rule runs the allen_swr_detector.py script
rule allen_swr_detector:
    priority: 3  # This rule has high priority
    conda:
        "allensdk_env.yaml"  # This is the Conda environment for the allen_swr_detector.py script
    shell:
        """
        echo "python allen_swr_detector.py --gamma_filter_path {config[gamma_filter_path]}"
        python allen_swr_detector.py 
        --pool_size {config[pool_size]} 
        --sdk_cache_dir {config[sdk_cache_dir]} 
        --output_dir {config[output_dir]} 
        --swr_output_dir {config[swr_output_dir]} 
        --run_name {config[run_name]} 
        --select_these_sessions {config[select_these_sessions]} 
        --only_brain_observatory_sessions {config[only_brain_observatory_sessions]} 
        --dont_wipe_these_sessions {config[dont_wipe_these_sessions]} 
        --gamma_event_thresh {config[gamma_event_thresh]} 
        --gamma_filter_path {config[gamma_filter_path]} 
        --theta_filter_path {config[theta_filter_path]} 
        --ripple_band_threshold {config[ripple_band_threshold]} 
        --movement_artifact_ripple_band_threshold {config[movement_artifact_ripple_band_threshold]} 
        > {output}
        """  # This is the command to run the allen_swr_detector.py script


# This rule runs the Filtering_SWR_Events_karlsson_detector.py script
rule Filtering_SWR_Events_karlsson_detector:
    priority: 2  # This rule has high priority
    conda:
        "allensdk_env.yaml"  # This is the Conda environment for the Filtering_SWR_Events_karlsson_detector.py script
    shell:
        """
        python Filtering_SWR_Events_karlsson_detector.py 
        --sdk_cache_dir_filter {config[sdk_cache_dir_filter]} 
        --input_dir {config[input_dir]} 
        --output_dir_filter {config[output_dir_filter]} 
        --swr_output_dir {config[swr_output_dir]} 
        {input} > {output}
        """  # This is the command to run the Filtering_SWR_Events_karlsson_detector.py script

# This rule runs the script with the new parameters
rule run_with_new_parameters:
    priority: 1  # This rule has high priority
    conda:
        "allensdk_env.yaml"  # This is the Conda environment for your script
    shell:
        """
        python your_script.py 
        --input_dir {config[input_dir]} 
        --output_dir {config[output_dir]} 
        --global_rip_label {config[global_rip_label]} 
        --minimum_ripple_num {config[minimum_ripple_num]} 
        {input} > {output}
        """  # This is the command to run your script