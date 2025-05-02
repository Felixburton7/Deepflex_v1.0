# mdcath_sampling
mdcath_sampling_pipeline/
├── holdout_selector/
│   ├── cath_parser.py             # Parses CATH file
│   ├── feature_extractor.py       # Loads features, calculates domain stats
│   ├── main_holdout_selector.py   # Orchestrates holdout selection (likely your old main.py/mdcath-holdout-extraction.py)
│   ├── sampling.py                # Core stratified sampling logic
│   ├── utils.py                   # Parallel processing helper
│   ├── validation.py              # Validation logic for sampling
│   ├── inputs/
│   │   └── cath-domain-list.txt   # Input CATH classification file (keep inputs separate)
│   └── outputs/                   # Generated domain lists
│       ├── mdcath_holdout_domains.txt
│       └── mdcath_training_domains.txt
│   └── README.md                  # Explains holdout selection process & how to run
│
├── dataset_creator/
│   ├── merge_rmsf_data.py         # Optional: Merges external predictions first
│   ├── split_features.py          # New script needed: Reads domain lists & feature CSVs, filters, saves subsets
│   ├── main_dataset_creator.py    # New script needed: Orchestrates merging (optional) and splitting
│   └── outputs/                   # Final datasets for ML
│       ├── holdout/               # Contains filtered CSVs for holdout domains
│       ├── train_base_models/     # Contains filtered CSVs for base model training domains
│       └── train_ensemble_model/  # Contains filtered CSVs for ensemble model training domains
│   └── README.md                  # Explains dataset creation process & how to run
│
└── README.md                      # Top-level README explaining the two parts and workflow