def parse_cath_file(file_path):
    """Parse CATH domain list file and return mapping dictionary."""
    cath_dict = {}
    with open(file_path, 'r') as f:
        for line in f:
            # Skip header lines and empty lines
            if line.startswith('#') or not line.strip():
                continue
                
            parts = line.split()
            if len(parts) >= 5:  # Ensure line has enough fields
                domain_id = parts[0]
                # Extract C.A.T.H values (columns 2-5)
                cath_class = int(parts[1])
                arch = int(parts[2])
                topo = int(parts[3])
                homo = int(parts[4])
                
                # Store as a tuple for easy access
                cath_dict[domain_id] = (cath_class, arch, topo, homo)
    
    print(f"Parsed {len(cath_dict)} domain classifications")
    return cath_dict