# #!/usr/bin/env python3
# """
# mdcath-holdout-extraction.py

# This script splits the mdCATH dataset into training and holdout sets based on
# a predefined list of holdout domains. Each temperature-specific dataset is split
# into separate files with no overlapping domains between training and holdout sets.
# """

# import os
# import pandas as pd

# def split_train_holdout_domains():
#     """
#     Split mdCATH data into training and holdout sets based on the holdout domains list.
#     Creates two new folders with separated datasets ensuring no domain crossover.
#     """
#     # Define file paths
#     base_dir = '/home/s_felix/mdcath-processor/outputs'
#     ml_features_dir = os.path.join(base_dir, 'ML_features')
#     holdout_list_path = '/home/s_felix/mdcath_sampling/mdcath_holdout_domains.txt'
    
#     # Define output directories
#     train_output_dir = os.path.join(base_dir, 'ML_features_train')
#     holdout_output_dir = os.path.join(base_dir, 'ML_features_holdout')
    
#     # Create output directories if they don't exist
#     os.makedirs(train_output_dir, exist_ok=True)
#     os.makedirs(holdout_output_dir, exist_ok=True)
    
#     # Load holdout domains into a set for faster lookup
#     with open(holdout_list_path, 'r') as f:
#         holdout_domains = set(line.strip() for line in f if line.strip())
    
#     print(f"Loaded {len(holdout_domains)} domains from holdout list")
    
#     # Process each CSV file in the ML_features directory
#     csv_files = [f for f in os.listdir(ml_features_dir) if f.endswith('.csv') and f.startswith('final_dataset_temperature_')]
    
#     for csv_file in csv_files:
#         # Extract temperature from filename
#         if '_temperature_' in csv_file:
#             temp = csv_file.split('_temperature_')[-1].replace('.csv', '')
#         else:
#             continue
        
#         csv_path = os.path.join(ml_features_dir, csv_file)
#         print(f"Processing {csv_path}...")
        
#         # Read the CSV file
#         df = pd.read_csv(csv_path)
        
#         # Get unique domains in this file
#         unique_domains = df['domain_id'].unique()
#         print(f"  - File contains {len(unique_domains)} unique domains")
        
#         # Create mask for holdout rows
#         is_holdout = df['domain_id'].isin(holdout_domains)
        
#         # Split the dataframe
#         holdout_df = df[is_holdout].copy()
#         train_df = df[~is_holdout].copy()
        
#         # Define output filenames with clean, consistent naming
#         holdout_output_file = f'temperature_{temp}_holdout.csv'
#         train_output_file = f'temperature_{temp}_train.csv'
        
#         holdout_output_path = os.path.join(holdout_output_dir, holdout_output_file)
#         train_output_path = os.path.join(train_output_dir, train_output_file)
        
#         # Save split datasets
#         holdout_df.to_csv(holdout_output_path, index=False)
#         train_df.to_csv(train_output_path, index=False)
        
#         # Get counts of unique domains in each set for verification
#         train_domains = train_df['domain_id'].unique()
#         holdout_domains_in_file = holdout_df['domain_id'].unique()
        
#         print(f"  - Split into:")
#         print(f"    * Training: {len(train_df)} rows, {len(train_domains)} unique domains → {train_output_path}")
#         print(f"    * Holdout: {len(holdout_df)} rows, {len(holdout_domains_in_file)} unique domains → {holdout_output_path}")
        
#         # Verify no overlap between sets
#         overlap = set(train_domains).intersection(set(holdout_domains_in_file))
#         if overlap:
#             print(f"  ⚠️ WARNING: Found {len(overlap)} domains in both training and holdout sets!")
#         else:
#             print(f"  ✓ Verified: No domain overlap between training and holdout sets")
    
#     print("\nSplit complete! Files are organized as follows:")
#     print(f"  - Training data: {train_output_dir}/")
#     print(f"  - Holdout data: {holdout_output_dir}/")

# if __name__ == "__main__":
#     split_train_holdout_domains()

#!/usr/bin/env python3
"""
mdcath-holdout-extraction.py

This script splits the mdCATH dataset into training and holdout sets based on
a predefined list of holdout domains. Each temperature-specific dataset is split
into separate files with no overlapping domains between training and holdout sets.
"""

import os
import pandas as pd
import glob

def split_train_holdout_domains():
    """
    Split mdCATH data into training and holdout sets based on the holdout domains list.
    Creates two new folders with separated datasets ensuring no domain crossover.
    """
    # Define file paths
    base_dir = '/home/s_felix/mdcath-processor/outputs'
    holdout_list_path = '/home/s_felix/mdcath_sampling/mdcath_holdout_domains.txt'
    
    # Define output directories
    train_output_dir = os.path.join(base_dir, 'RMSF_replica_average_train')
    holdout_output_dir = os.path.join(base_dir, 'RMSF_replica_average_holdout')
    
    # Create output directories if they don't exist
    os.makedirs(train_output_dir, exist_ok=True)
    os.makedirs(holdout_output_dir, exist_ok=True)
    
    # Load holdout domains into a set for faster lookup
    with open(holdout_list_path, 'r') as f:
        holdout_domains = set(line.strip() for line in f if line.strip())
    
    print(f"Loaded {len(holdout_domains)} domains from holdout list")
    
    # Find all RMSF replica average files using glob pattern
    # This will search for the specific file pattern in all temperature subdirectories
    rmsf_files = glob.glob(os.path.join(base_dir, 'RMSF/replica_average/**/rmsf_replica_average_temperature*.csv'), recursive=True)
    
    if not rmsf_files:
        print("No RMSF replica average files found!")
        return
    
    print(f"Found {len(rmsf_files)} RMSF replica average files")
    
    for file_path in rmsf_files:
        # Extract temperature from filename
        filename = os.path.basename(file_path)
        
        # Extract temperature from the filename
        # Assuming format like "rmsf_replica_average_temperature320.csv"
        if 'temperature' in filename:
            temp = filename.split('temperature')[-1].split('.')[0]
        else:
            print(f"  - Skipping {filename} - unable to extract temperature")
            continue
        
        print(f"Processing {file_path} (Temperature: {temp})...")
        
        try:
            # Read the CSV file
            df = pd.read_csv(file_path)
            
            # Get unique domains in this file
            unique_domains = df['domain_id'].unique()
            print(f"  - File contains {len(unique_domains)} unique domains")
            
            # Create mask for holdout rows
            is_holdout = df['domain_id'].isin(holdout_domains)
            
            # Split the dataframe
            holdout_df = df[is_holdout].copy()
            train_df = df[~is_holdout].copy()
            
            # Define output filenames with clean, consistent naming
            holdout_output_file = f'rmsf_temperature_{temp}_holdout.csv'
            train_output_file = f'rmsf_temperature_{temp}_train.csv'
            
            holdout_output_path = os.path.join(holdout_output_dir, holdout_output_file)
            train_output_path = os.path.join(train_output_dir, train_output_file)
            
            # Save split datasets
            holdout_df.to_csv(holdout_output_path, index=False)
            train_df.to_csv(train_output_path, index=False)
            
            # Get counts of unique domains in each set for verification
            train_domains = train_df['domain_id'].unique()
            holdout_domains_in_file = holdout_df['domain_id'].unique()
            
            print(f"  - Split into:")
            print(f"    * Training: {len(train_df)} rows, {len(train_domains)} unique domains → {train_output_path}")
            print(f"    * Holdout: {len(holdout_df)} rows, {len(holdout_domains_in_file)} unique domains → {holdout_output_path}")
            
            # Verify no overlap between sets
            overlap = set(train_domains).intersection(set(holdout_domains_in_file))
            if overlap:
                print(f"  ⚠️ WARNING: Found {len(overlap)} domains in both training and holdout sets!")
            else:
                print(f"  ✓ Verified: No domain overlap between training and holdout sets")
                
        except Exception as e:
            print(f"  - Error processing {file_path}: {str(e)}")
    
    print("\nSplit complete! Files are organized as follows:")
    print(f"  - Training data: {train_output_dir}/")
    print(f"  - Holdout data: {holdout_output_dir}/")

if __name__ == "__main__":
    split_train_holdout_domains()