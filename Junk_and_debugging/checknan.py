import os
import gzip
import csv

BASE_DIR = "/space/scratch/SWR_final_pipeline/osf_campbellmurphy2025_swr_data_backup"
FOLDERS = [
    "allen_visbehave_swr_murphylab2024",
    "ibl_swr_murphylab2024",
    "allen_viscoding_swr_murphylab2024",
]

def process_folder(folder_path):
    total_rows = 0
    nan_rows = 0
    # Walk through all subdirs in folder_path
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith("_putative_swr_events.csv.gz"):
                file_path = os.path.join(root, file)
                with gzip.open(file_path, mode='rt', newline='') as f:
                    reader = csv.reader(f)
                    # Skip header
                    try:
                        header = next(reader)
                    except StopIteration:
                        # Empty file, skip
                        continue

                    for row in reader:
                        total_rows += 1
                        # Check if any column is empty or whitespace-only (considered NaN)
                        if any(cell.strip() == "" for cell in row):
                            nan_rows += 1
    return total_rows, nan_rows

def main():
    for folder in FOLDERS:
        folder_path = os.path.join(BASE_DIR, folder)
        print(f"Processing: {folder}")
        total, nan = process_folder(folder_path)
        print(f"Total rows: {total}")
        print(f"Rows with NaNs: {nan}\n")

if __name__ == "__main__":
    main()
