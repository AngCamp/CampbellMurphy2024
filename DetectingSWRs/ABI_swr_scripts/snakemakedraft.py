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
        "python Filtering_SWR_Events_karlsson_detector.py {input} > {output}"  # This is the command to run the Filtering_SWR_Events_karlsson_detector.py script

# This rule runs the Global_Ripple_detector.py script
rule Global_Ripple_detector:
    conda:
        "envs/Global_Ripple_detector.yaml"  # This is the Conda environment for the Global_Ripple_detector.py script
    input:
        rules.Filtering_SWR_Events_karlsson_detector.output  # This is the input file for the Global_Ripple_detector.py script, which is the output file from the Filtering_SWR_Events_karlsson_detector.py script
    output:
        "output/Global_Ripple_detector_output.txt"  # This is the output file from the Global_Ripple_detector.py script
    shell:
        "python Global_Ripple_detector.py {input} > {output}"  # This is the command to run the Global_Ripple_detector.py script