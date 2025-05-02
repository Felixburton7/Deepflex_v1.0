import numpy as np
from scipy import stats
from collections import Counter, defaultdict

def statistical_validation(holdout_domains, all_domains, domain_features, cath_dict):
    """Perform statistical validation of the holdout set."""
    validation_results = {}
    
    # Filter for domains with features
    holdout_with_features = [d for d in holdout_domains if d in domain_features]
    all_with_features = [d for d in all_domains if d in domain_features]
    
    # 1. Domain size distribution (KS test)
    holdout_sizes = [domain_features[d]['size'] for d in holdout_with_features]
    all_sizes = [domain_features[d]['size'] for d in all_with_features]
    
    ks_size_stat, ks_size_pval = stats.kstest(holdout_sizes, all_sizes)
    validation_results['ks_size_pval'] = ks_size_pval
    
    # 2. RMSF distributions (KS test for 320K)
    if 'avg_rmsf_320' in domain_features.get(holdout_with_features[0], {}):
        holdout_rmsf = [domain_features[d]['avg_rmsf_320'] for d in holdout_with_features 
                        if 'avg_rmsf_320' in domain_features[d]]
        all_rmsf = [domain_features[d]['avg_rmsf_320'] for d in all_with_features 
                   if 'avg_rmsf_320' in domain_features[d]]
        
        ks_rmsf_stat, ks_rmsf_pval = stats.kstest(holdout_rmsf, all_rmsf)
        validation_results['ks_rmsf_pval'] = ks_rmsf_pval
    
    # 3. Secondary structure composition
    holdout_helix = [domain_features[d]['helix_pct'] for d in holdout_with_features]
    all_helix = [domain_features[d]['helix_pct'] for d in all_with_features]
    
    holdout_sheet = [domain_features[d]['sheet_pct'] for d in holdout_with_features]
    all_sheet = [domain_features[d]['sheet_pct'] for d in all_with_features]
    
    ks_helix_stat, ks_helix_pval = stats.kstest(holdout_helix, all_helix)
    ks_sheet_stat, ks_sheet_pval = stats.kstest(holdout_sheet, all_sheet)
    
    validation_results['ks_helix_pval'] = ks_helix_pval
    validation_results['ks_sheet_pval'] = ks_sheet_pval
    
    # 4. CATH classification counts
    holdout_cath = [cath_dict[d] for d in holdout_domains if d in cath_dict]
    all_cath = [cath_dict[d] for d in all_domains if d in cath_dict]
    
    # Class level (C)
    holdout_class = [c[0] for c in holdout_cath]
    all_class = [c[0] for c in all_cath]
    
    holdout_class_count = Counter(holdout_class)
    all_class_count = Counter(all_class)
    
    # Calculate expected frequencies
    expected = {}
    for c in all_class_count:
        expected[c] = all_class_count[c] * len(holdout_class) / len(all_class)
    
    # Chi-squared test if possible
    if len(holdout_class_count) > 1 and len(all_class_count) > 1:
        chi2 = 0
        for c in all_class_count:
            observed = holdout_class_count.get(c, 0)
            if expected[c] > 0:
                chi2 += ((observed - expected[c]) ** 2) / expected[c]
        
        df = len(all_class_count) - 1
        chi2_pval = 1 - stats.chi2.cdf(chi2, df)
        validation_results['chi2_class_pval'] = chi2_pval
    
    # 5. Stability class representation
    if 'stability_class' in domain_features.get(holdout_with_features[0], {}):
        holdout_stability = [domain_features[d].get('stability_class', 'unknown') for d in holdout_with_features]
        all_stability = [domain_features[d].get('stability_class', 'unknown') for d in all_with_features]
        
        holdout_stability_count = Counter(holdout_stability)
        all_stability_count = Counter(all_stability)
        
        # Calculate stability profile coverage
        stability_classes = set(all_stability_count.keys())
        represented_classes = set(holdout_stability_count.keys())
        
        stability_coverage = len(represented_classes) / len(stability_classes) if stability_classes else 1.0
        validation_results['stability_coverage'] = stability_coverage
    
    # Calculate hierarchy coverage
    all_c = set(c[0] for c in all_cath)
    all_ca = set((c[0], c[1]) for c in all_cath)
    all_cat = set((c[0], c[1], c[2]) for c in all_cath)
    
    holdout_c = set(c[0] for c in holdout_cath)
    holdout_ca = set((c[0], c[1]) for c in holdout_cath)
    holdout_cat = set((c[0], c[1], c[2]) for c in holdout_cath)
    
    c_coverage = len(holdout_c) / len(all_c) if all_c else 1.0
    ca_coverage = len(holdout_ca) / len(all_ca) if all_ca else 1.0
    cat_coverage = len(holdout_cat) / len(all_cat) if all_cat else 1.0
    
    # Weighted average of hierarchy coverage
    hierarchy_coverage = (0.2 * c_coverage + 0.3 * ca_coverage + 0.5 * cat_coverage)
    validation_results['hierarchy_coverage'] = hierarchy_coverage
    
    # Calculate distribution similarity (average p-value)
    p_values = [v for k, v in validation_results.items() if k.endswith('_pval')]
    distribution_similarity = np.mean(p_values) if p_values else 0.0
    validation_results['distribution_similarity'] = distribution_similarity
    
    # Calculate Representation Index
    stability_coverage = validation_results.get('stability_coverage', 1.0)
    ri = (distribution_similarity * hierarchy_coverage * stability_coverage) ** (1/3)
    validation_results['representation_index'] = ri
    
    return validation_results

def calculate_representation_index(validation_results):
    """Calculate Representation Index from validation results."""
    distribution_similarity = validation_results.get('distribution_similarity', 0)
    hierarchy_coverage = validation_results.get('hierarchy_coverage', 0)
    stability_coverage = validation_results.get('stability_coverage', 0)
    
    ri = (distribution_similarity * hierarchy_coverage * stability_coverage) ** (1/3)
    return ri

