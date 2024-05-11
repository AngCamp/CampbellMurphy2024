# Load parameters from config file
configfile: "config.yaml"


# This is the final rule that requires the output from the Global_Ripple_detector rule
rule all:
    input:
        "output/Global_Ripple_detector_output.txt"  # This is the final output file we want to generate

# This rule runs the allen_swr_detector.py script
rule allen_swr_detector:
    conda:
        "envs/allen_swr_detector.yaml"  # This is the Conda environment for the allen_swr_detector.py script
    output:
        "output/allen_swr_detector_output.txt"  # This is the output file from the allen_swr_detector.py script
    shell:
        "python allen_swr_detector.py > {output}"  # This is the command to run the allen_swr_detector.py script


# This rule runs the Filtering_SWR_Events_karlsson_detector.py script
rule Filtering_SWR_Events_karlsson_detector:
    conda:
        "envs/Filtering_SWR_Events_karlsson_detector.yaml"  # This is the Conda environment for the Filtering_SWR_Events_karlsson_detector.py script
    input:
        rules.allen_swr_detector.output  # This is the input file for the Filtering_SWR_Events_karlsson_detector.py script, which is the output file from the allen_swr_detector.py script
    output:
        "output/Filtering_SWR_Events_karlsson_detector_output.txt"  # This is the output file from the Filtering_SWR_Events_karlsson_detector.py script
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
    conda:
        "envs/your_script_environment.yaml"  # This is the Conda environment for your script
    input:
        rules.Filtering_SWR_Events_karlsson_detector.output  # This is the input file for your script, which is the output file from the Filtering_SWR_Events_karlsson_detector.py script
    output:
        "output/your_script_output.txt"  # This is the output file from your script
    shell:
        """
        python your_script.py 
        --input_dir {config[input_dir]} 
        --output_dir {config[output_dir]} 
        --global_rip_label {config[global_rip_label]} 
        --minimum_ripple_num {config[minimum_ripple_num]} 
        {input} > {output}
        """  # This is the command to run your script