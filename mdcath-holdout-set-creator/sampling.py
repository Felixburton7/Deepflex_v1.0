import numpy as np
from collections import defaultdict
from sklearn.cluster import KMeans
import networkx as nx

def hierarchical_stratified_sampling(domain_features, cath_dict, sample_ratio=0.1):
    """Perform hierarchical stratified sampling based on CATH classification."""
    # Organize domains by CATH hierarchy
    cath_hierarchy = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    # Only include domains that are in both the feature set and CATH dictionary
    domains = set(domain_features.keys()).intersection(set(cath_dict.keys()))
    
    for domain in domains:
        if domain in cath_dict:
            c, a, t, h = cath_dict[domain]
            cath_hierarchy[c][a][t].append(domain)
    
    # Selected domains for holdout
    holdout_domains = []
    
    # Process each level of the hierarchy
    for c_class, architectures in cath_hierarchy.items():
        for a_arch, topologies in architectures.items():
            for t_topo, domain_list in topologies.items():
                if len(domain_list) >= 10:
                    # For larger topologies, use k-means clustering
                    n_clusters = max(1, int(np.ceil(len(domain_list) * sample_ratio)))
                    
                    # Prepare feature matrix for clustering
                    feature_matrix = []
                    domain_order = []
                    
                    for domain in domain_list:
                        if domain in domain_features:
                            # Extract relevant features for clustering
                            features = [
                                domain_features[domain].get('size', 0),
                                domain_features[domain].get('helix_pct', 0),
                                domain_features[domain].get('sheet_pct', 0),
                                domain_features[domain].get('core_ratio', 0),
                                domain_features[domain].get('avg_accessibility', 0)
                            ]
                            
                            # Add RMSF features if available
                            if 'avg_rmsf_320' in domain_features[domain]:
                                features.append(domain_features[domain]['avg_rmsf_320'])
                                
                            feature_matrix.append(features)
                            domain_order.append(domain)
                    
                    if not feature_matrix:  # Skip if no features available
                        continue
                        
                    # Normalize features
                    feature_matrix = np.array(feature_matrix)
                    feature_matrix = (feature_matrix - np.mean(feature_matrix, axis=0)) / (np.std(feature_matrix, axis=0) + 1e-10)
                    
                    # Perform k-means clustering
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                    clusters = kmeans.fit_predict(feature_matrix)
                    
                    # Select one domain from each cluster
                    for cluster_id in range(n_clusters):
                        cluster_domains = [domain_order[i] for i, c in enumerate(clusters) if c == cluster_id]
                        if cluster_domains:
                            # Select domain closest to cluster center
                            holdout_domains.append(np.random.choice(cluster_domains))
                
                elif len(domain_list) > 0:
                    # For smaller topologies, sample with probability
                    if len(domain_list) == 1:
                        # Single domain topology, include with 10% probability
                        if np.random.random() < sample_ratio:
                            holdout_domains.append(domain_list[0])
                    else:
                        # Multiple domains but <10, sample one with probability
                        num_to_sample = max(1, int(len(domain_list) * sample_ratio))
                        sampled = np.random.choice(domain_list, num_to_sample, replace=False)
                        holdout_domains.extend(sampled)
    
    # Network-based homology control
    holdout_domains = network_aware_sampling(holdout_domains, domain_features, cath_dict, sample_ratio)
    
    return holdout_domains

def network_aware_sampling(candidate_domains, domain_features, cath_dict, sample_ratio=0.1):
    """Apply network-based homology control to sampling."""
    # Build domain network
    G = nx.Graph()
    
    # Add all domains as nodes
    for domain in candidate_domains:
        G.add_node(domain)
    
    # Connect domains if they share PDB ID or homology
    for i, domain1 in enumerate(candidate_domains):
        pdb_id1 = domain1[:4]
        
        for j in range(i+1, len(candidate_domains)):
            domain2 = candidate_domains[j]
            pdb_id2 = domain2[:4]
            
            # Connect if same PDB ID
            if pdb_id1 == pdb_id2:
                G.add_edge(domain1, domain2)
            
            # Connect if same H-level (homologous superfamily)
            elif domain1 in cath_dict and domain2 in cath_dict:
                c1, a1, t1, h1 = cath_dict[domain1]
                c2, a2, t2, h2 = cath_dict[domain2]
                
                if c1 == c2 and a1 == a2 and t1 == t2 and h1 == h2:
                    G.add_edge(domain1, domain2)
    
    # Find connected components
    connected_components = list(nx.connected_components(G))
    
    # Sample from connected components
    final_holdout = []
    num_components_to_sample = max(1, int(len(connected_components) * sample_ratio))
    
    # Sort components by size to ensure representation of larger components
    sorted_components = sorted(connected_components, key=len, reverse=True)
    
    # Stratify by stability class if possible
    stability_components = defaultdict(list)
    
    for i, component in enumerate(sorted_components):
        # Determine majority stability class for this component
        stability_counts = defaultdict(int)
        for domain in component:
            if domain in domain_features:
                stability_class = domain_features[domain].get('stability_class', 'unknown')
                stability_counts[stability_class] += 1
        
        # Assign to majority class
        if stability_counts:
            majority_class = max(stability_counts.items(), key=lambda x: x[1])[0]
            stability_components[majority_class].append((i, component))
        else:
            stability_components['unknown'].append((i, component))
    
    # Sample from each stability class
    sampled_indices = []
    
    for stability_class, components in stability_components.items():
        class_sample_size = max(1, int(len(components) * sample_ratio))
        selected_indices = np.random.choice(len(components), class_sample_size, replace=False)
        
        for idx in selected_indices:
            sampled_indices.append(components[idx][0])
    
    # Add all domains from selected components
    for idx in sorted(sampled_indices):
        component = sorted_components[idx]
        final_holdout.extend(component)
    
    return final_holdout