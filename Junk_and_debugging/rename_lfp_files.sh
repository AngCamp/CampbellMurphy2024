#!/bin/bash

# Base directory for the data
BASE_DIR="/space/scratch/SWR_final_pipeline/osf_campbellmurphy2025_swr_data"

# Process each dataset's LFP directory
for dataset in "allen_visbehave_swr_murphylab2024_lfp_data" "allen_viscoding_swr_murphylab2024_lfp_data" "ibl_swr_murphylab2024_lfp_data"; do
    echo "Processing dataset: $dataset"
    
    # Find all session directories
    for session_dir in "$BASE_DIR/$dataset"/lfp_session_*; do
        if [ -d "$session_dir" ]; then
            echo "Processing session: $session_dir"
            
            # Rename ca1_peakripplepower.npz files
            for file in "$session_dir"/probe_*_channel_*_lfp_ca1_peakripplepower.npz; do
                if [ -f "$file" ]; then
                    new_name="${file/ca1_peakripplepower.npz/ca1_putative_pyramidal_layer.npz}"
                    mv "$file" "$new_name"
                    echo "Renamed: $file -> $new_name"
                fi
            done
            
            # Rename ca1_sharpwave.npz files
            for file in "$session_dir"/probe_*_channel_*_lfp_ca1_sharpwave.npz; do
                if [ -f "$file" ]; then
                    new_name="${file/ca1_sharpwave.npz/ca1_putative_str_radiatum.npz}"
                    mv "$file" "$new_name"
                    echo "Renamed: $file -> $new_name"
                fi
            done

            # Rename control channel files to remove channelsrawInd_ prefix
            for file in "$session_dir"/probe_*_channel_channelsrawInd_*_lfp_control_channel.npz; do
                if [ -f "$file" ]; then
                    new_name="${file//channelsrawInd_/}"
                    mv "$file" "$new_name"
                    echo "Renamed: $file -> $new_name"
                fi
            done
        fi
    done
done

# 2. Rename movement artifact CSVs in SWR session folders
for dataset in "allen_visbehave_swr_murphylab2024" "allen_viscoding_swr_murphylab2024" "ibl_swr_murphylab2024"; do
    echo "Processing SWR dataset: $dataset"
    for session_dir in "$BASE_DIR/$dataset"/swrs_session_*; do
        if [ -d "$session_dir" ]; then
            for file in "$session_dir"/probe_*_channel_channelsrawInd_*_movement_artifacts.csv.gz; do
                if [ -f "$file" ]; then
                    new_name="${file//channelsrawInd_/}"
                    mv "$file" "$new_name"
                    echo "Renamed CSV: $file -> $new_name"
                fi
            done
        fi
    done
done

echo "All control channel renaming complete!" 