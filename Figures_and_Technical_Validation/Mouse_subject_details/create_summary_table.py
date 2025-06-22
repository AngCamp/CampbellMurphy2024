import argparse
import pandas as pd
import numpy as np
import os

def create_abi_summary_table(df, output_file):
    """Creates a summary demographics table for ABI datasets."""
    df['age_weeks'] = df['age_in_days'] / 7
    mouse_id_col = 'specimen_id'

    # Ensure 'full_genotype' exists and handle potential NaN values
    if 'full_genotype' not in df.columns:
        df['full_genotype'] = 'N/A'
    df['full_genotype'] = df['full_genotype'].fillna('N/A')


    mouse_stats = df.groupby([mouse_id_col, 'full_genotype', 'sex']).agg(
        age_weeks=('age_weeks', 'mean'),
        probes_in_ca1=('probes_in_ca1', 'mean')
    ).reset_index()

    summary = mouse_stats.groupby(['full_genotype', 'sex']).agg(
        Count=(mouse_id_col, 'nunique'),
        Avg_Age_Weeks=('age_weeks', 'mean'),
        Min_Age_Weeks=('age_weeks', 'min'),
        Max_Age_Weeks=('age_weeks', 'max'),
        Avg_Probes_in_CA1=('probes_in_ca1', 'mean')
    ).reset_index()

    pivot_summary = summary.pivot(index='full_genotype', columns='sex', values=['Count', 'Avg_Age_Weeks', 'Min_Age_Weeks', 'Max_Age_Weeks', 'Avg_Probes_in_CA1'])
    final_table = pd.DataFrame(index=pivot_summary.index)
    
    for sex in ['M', 'F']:
        sex_label = 'Male' if sex == 'M' else 'Female'
        if ('Count', sex) in pivot_summary.columns:
            final_table[f'Count ({sex_label})'] = pivot_summary[('Count', sex)].fillna(0).astype(int)
        else:
            final_table[f'Count ({sex_label})'] = 0

    for sex in ['M', 'F']:
        sex_label = 'Male' if sex == 'M' else 'Female'
        if ('Avg_Age_Weeks', sex) in pivot_summary.columns:
            final_table[f'Avg Age (Weeks) ({sex_label})'] = pivot_summary[('Avg_Age_Weeks', sex)].round(2)
        else:
            final_table[f'Avg Age (Weeks) ({sex_label})'] = np.nan
            
    for sex in ['M', 'F']:
        sex_label = 'Male' if sex == 'M' else 'Female'
        if ('Min_Age_Weeks', sex) in pivot_summary.columns and ('Max_Age_Weeks', sex) in pivot_summary.columns:
            final_table[f'Age Range (Weeks) ({sex_label})'] = \
                pivot_summary[('Min_Age_Weeks', sex)].round(0).astype(int).astype(str) + '-' + \
                pivot_summary[('Max_Age_Weeks', sex)].round(0).astype(int).astype(str)
        else:
            final_table[f'Age Range (Weeks) ({sex_label})'] = 'N/A'

    for sex in ['M', 'F']:
        sex_label = 'Male' if sex == 'M' else 'Female'
        if ('Avg_Probes_in_CA1', sex) in pivot_summary.columns:
            final_table[f'Avg Probes in CA1 ({sex_label})'] = pivot_summary[('Avg_Probes_in_CA1', sex)].round(2)
        else:
            final_table[f'Avg Probes in CA1 ({sex_label})'] = np.nan

    final_table = final_table[[
        'Count (Male)', 'Count (Female)',
        'Avg Age (Weeks) (Male)', 'Avg Age (Weeks) (Female)',
        'Age Range (Weeks) (Male)', 'Age Range (Weeks) (Female)',
        'Avg Probes in CA1 (Male)', 'Avg Probes in CA1 (Female)'
    ]]
    
    print_and_save_table(final_table, output_file, "ABI Summary Table")

def create_ibl_summary_table(df, output_file):
    """Creates a simplified summary table for the IBL dataset."""
    mouse_stats = df.groupby(['subject_id', 'sex']).agg(
        age_weeks=('age_weeks', 'mean'),
        probes_in_ca1=('probes_in_ca1', 'mean')
    ).reset_index()

    summary = mouse_stats.groupby('sex').agg(
        Count=('subject_id', 'nunique'),
        Avg_Age_Weeks=('age_weeks', 'mean'),
        Min_Age_Weeks=('age_weeks', 'min'),
        Max_Age_Weeks=('age_weeks', 'max'),
        Avg_Probes_in_CA1=('probes_in_ca1', 'mean')
    ).reset_index()

    summary['Avg Age (Weeks)'] = summary['Avg_Age_Weeks'].round(1)
    summary['Age Range (Weeks)'] = summary.apply(
        lambda row: f"{int(row['Min_Age_Weeks'])}-{int(row['Max_Age_Weeks'])}", axis=1
    )
    summary['Avg Probes in CA1'] = summary['Avg_Probes_in_CA1'].round(2)
    summary['Sex'] = summary['sex'].map({'M': 'Male', 'F': 'Female'})

    total_count = mouse_stats['subject_id'].nunique()
    total_avg_age = mouse_stats['age_weeks'].mean()
    total_min_age = mouse_stats['age_weeks'].min()
    total_max_age = mouse_stats['age_weeks'].max()
    total_avg_probes = mouse_stats['probes_in_ca1'].mean()

    total_row = pd.DataFrame([{
        'Sex': 'Total',
        'Count': total_count,
        'Avg Age (Weeks)': round(total_avg_age, 1),
        'Age Range (Weeks)': f"{int(total_min_age)}-{int(total_max_age)}",
        'Avg Probes in CA1': round(total_avg_probes, 2)
    }])
    
    final_table = summary[['Sex', 'Count', 'Avg Age (Weeks)', 'Age Range (Weeks)', 'Avg Probes in CA1']]
    final_table = pd.concat([final_table, total_row], ignore_index=True)
    
    print_and_save_table(final_table, output_file, "IBL Summary Table")

def print_and_save_table(df, output_file, title):
    """Prints and saves the final dataframe."""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"\n--- {title} ---")
    print(df.to_string())
    print("\n---------------------\n")
    print(f"Summary table saved to {output_file}")


def main(dataset_name):
    """
    Creates a summary subject information table from a raw subject info CSV.
    """
    file_map = {
        'abi_visual_behaviour': 'visbehaviour_subject_info.csv',
        'abi_visual_coding': 'viscoding_subject_info.csv',
        'ibl': 'ibl_subject_info.csv'
    }

    if dataset_name not in file_map:
        print(f"Error: Invalid dataset name '{dataset_name}'.")
        print(f"Choose from: {', '.join(file_map.keys())}")
        return

    input_filename = file_map[dataset_name]
    input_filepath = os.path.join('subject_info_data', input_filename)

    if not os.path.exists(input_filepath):
        print(f"Error: Input file not found at {input_filepath}")
        print("Please run the corresponding get_*_subject_info.py script first.")
        return

    output_filename = input_filename.replace('.csv', '_summary.csv')
    output_filepath = os.path.join('subject_info_summaries', output_filename)
    
    df = pd.read_csv(input_filepath)

    if dataset_name == 'ibl':
        create_ibl_summary_table(df, output_filepath)
    else:
        create_abi_summary_table(df, output_filepath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a summary subject information table from a previously generated data file.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        'dataset_name', 
        type=str, 
        choices=['abi_visual_behaviour', 'abi_visual_coding', 'ibl'],
        help="""Name of the dataset to process.
- abi_visual_behaviour: Allen Visual Behaviour dataset (Table 1)
- abi_visual_coding: Allen Visual Coding dataset (Table 2)
- ibl: IBL dataset (Table 3)"""
    )
    
    args = parser.parse_args()
    
    main(args.dataset_name)

    # Example usage from command line:
    # python create_summary_table.py visbehaviour_subject_info.csv --output_file visbehaviour_summary.csv
    # python create_summary_table.py viscoding_subject_info.csv --output_file viscoding_summary.csv
    # python create_summary_table.py ibl_subject_info.csv --output_file ibl_summary.csv --is_ibl 