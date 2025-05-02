import os
import sys
import time
import numpy as np
from cath_parser import parse_cath_file
from feature_extractor import load_temperature_data, aggregate_domain_features
from sampling import hierarchical_stratified_sampling
from validation import statistical_validation, calculate_representation_index

def main():
    print("Starting mdCATH Dataset Stratified Sampling...")
    
    # Parse CATH classifications
    print("Parsing CATH classifications...")
    cath_dict = parse_cath_file('/home/s_felix/mdcath_sampling/cath-domain-list.txt')
    
    # Load temperature data
    print("Loading temperature data...")
    file_paths = [
        'final_dataset_temperature_320.csv',
        'final_dataset_temperature_348.csv',
        'final_dataset_temperature_379.csv',
        'final_dataset_temperature_413.csv',
        'final_dataset_temperature_450.csv',
        'final_dataset_temperature_average.csv'
    ]
    
    # Update file paths to include full path
    file_paths = [os.path.join('/home/s_felix/mdcath-processor/outputs/ML_features', f) for f in file_paths]
    
    temperature_data = load_temperature_data(file_paths)
    
    # Compute domain-level features
    print("Computing domain-level features...")
    domain_features = aggregate_domain_features(temperature_data)
    
    # Get all domain IDs
    all_domains = list(domain_features.keys())
    print(f"Total domains with features: {len(all_domains)}")
    
    # Perform hierarchical stratified sampling
    print("Performing hierarchical stratified sampling...")
    holdout_domains = hierarchical_stratified_sampling(domain_features, cath_dict, sample_ratio=0.5)
    
    # Initial holdout size
    print(f"Initial holdout size: {len(holdout_domains)} domains")
    
    # Validate the holdout set
    print("Validating holdout set...")
    validation_results = statistical_validation(holdout_domains, all_domains, domain_features, cath_dict)
    ri = validation_results['representation_index']
    print(f"Representation Index: {ri:.4f}")
    
    # Iterative refinement if needed
    max_iterations = 5
    iteration = 0
    
    while ri < 0.9 and iteration < max_iterations:
        print(f"Representation Index {ri:.4f} < 0.9, refining sampling...")
        iteration += 1
        
        # Adjust sample ratio based on current results
        sample_ratio = 0.1 * (1 + 0.4 * iteration)

        
        # Repeat sampling
        holdout_domains = hierarchical_stratified_sampling(domain_features, cath_dict, sample_ratio=sample_ratio)
        
        # Validate again
        validation_results = statistical_validation(holdout_domains, all_domains, domain_features, cath_dict)
        ri = validation_results['representation_index']
        print(f"Iteration {iteration}, Representation Index: {ri:.4f}")
    
    # Final holdout size
    print(f"Final holdout size: {len(holdout_domains)} domains ({len(holdout_domains)/len(all_domains)*100:.1f}% of total)")
    
    # Save holdout domains to file
    output_file = 'mdcath_holdout_domains.txt'
    with open(output_file, 'w') as f:
        for domain in sorted(holdout_domains):
            f.write(f"{domain}\n")
    
    # Save training domains to file (all domains not in holdout)
    training_domains = list(set(all_domains) - set(holdout_domains))
    training_file = 'mdcath_training_domains.txt'
    with open(training_file, 'w') as f:
        for domain in sorted(training_domains):
            f.write(f"{domain}\n")
    
    print(f"Holdout domains saved to {output_file}")
    print(f"Training domains saved to {training_file}")
    print("Stratified sampling completed successfully!")

if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed_time = time.time() - start_time
    print(f"Total execution time: {elapsed_time:.2f} seconds")