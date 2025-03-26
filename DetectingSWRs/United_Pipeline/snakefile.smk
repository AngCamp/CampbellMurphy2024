# Snakefile for SWR detection pipeline - Sequential processing

# Configuration
configfile: "united_detector_config.yaml"

import os
import numpy as np

# Helper function to determine the appropriate conda environment
def get_conda_env(dataset):
    if dataset in ["abi_visual_behaviour", "abi_visual_coding"]:
        return "allensdk_env"
    elif dataset == "ibl":
        return "ONE_ibl_env"
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

# Get datasets from command line or use all datasets
datasets_param = config.get("datasets", "ibl,abi_visual_behaviour,abi_visual_coding")
DATASETS = datasets_param.split(",")

# Validate datasets
for dataset in DATASETS:
    if dataset not in ["ibl", "abi_visual_behaviour", "abi_visual_coding"]:
        raise ValueError(f"Unknown dataset: {dataset}")

# Define the final output that depends on all datasets being processed
rule all:
    input:
        # This creates a dependency chain that forces sequential processing
        "results/processing_complete.txt"

# This rule ensures sequential processing by creating a dependency chain
rule finalize:
    input:
        # Process the datasets one by one in the order specified
        expand("results/{dataset}/processing_complete.txt", dataset=DATASETS)
    output:
        "results/processing_complete.txt"
    shell:
        """
        echo "All datasets processed sequentially:" > {output}
        echo "{DATASETS}" >> {output}
        echo "Completed at $(date)" >> {output}
        """

rule process_dataset:
    input:
        config = "united_detector_config.yaml",
        # This ensures datasets are processed one after another
        # Each dataset depends on the previous one being finished
        prev_dataset = lambda wildcards: (
            "results/start_marker.txt" if DATASETS.index(wildcards.dataset) == 0 
            else f"results/{DATASETS[DATASETS.index(wildcards.dataset) - 1]}/processing_complete.txt"
        )
    output:
        marker = "results/{dataset}/processing_complete.txt"
    conda:
        lambda wildcards: get_conda_env(wildcards.dataset)
    params:
        dataset = "{dataset}"
    threads: lambda wildcards: config["pool_sizes"][wildcards.dataset]
    shell:
        """
        echo "Starting processing {params.dataset} with {threads} cores at $(date)"
        
        # Set environment variables
        export DATASET_TO_PROCESS={params.dataset}
        export POOL_SIZE={threads}
        
        # Create output directory
        mkdir -p results/{params.dataset}
        
        # Run the detector script with the dataset-specific pool size
        python united_swr_detector.py
        
        # Create marker file to indicate completion
        echo "Processing of {params.dataset} completed at $(date)" > {output.marker}
        """

# First rule to kick off the sequential processing
rule start:
    output:
        "results/start_marker.txt"
    shell:
        """
        mkdir -p results
        echo "Starting pipeline execution at $(date)" > {output}
        echo "Datasets to process (in order): {DATASETS}" >> {output}
        """

# Optional rule for generating a report
rule generate_report:
    input:
        "results/processing_complete.txt"
    output:
        report = "final_report.html"
    shell:
        """
        echo "<html><body>" > {output.report}
        echo "<h1>SWR Detection Pipeline Results</h1>" >> {output.report}
        echo "<p>Datasets processed sequentially in this order: {DATASETS}</p>" >> {output.report}
        
        for dataset in {DATASETS}; do
            echo "<h2>$dataset</h2>" >> {output.report}
            echo "<pre>" >> {output.report}
            cat results/$dataset/processing_complete.txt >> {output.report}
            echo "</pre>" >> {output.report}
        done
        
        echo "</body></html>" >> {output.report}
        """