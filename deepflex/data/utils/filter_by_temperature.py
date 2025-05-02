import pandas as pd
from pathlib import Path

# ==== Configuration ====
INPUT_CSV_PATH = Path("/home/s_felix/FINAL_PROJECT/packages/DeepFlex/data/raw/aggregated_train_dataset.csv")
OUTPUT_DIR = Path("/home/s_felix/FINAL_PROJECT/packages/DeepFlex/data/raw")
TEMPERATURES = [320, 348, 379, 413, 450]
BASENAME_TEMPLATE = "aggregated_train_{}k_dataset.csv"

# ==== Load Dataset ====
df = pd.read_csv(INPUT_CSV_PATH)

# ==== Filter and Save by Temperature ====
for temp in TEMPERATURES:
    filtered = df[df["temperature"] == temp]
    output_path = OUTPUT_DIR / BASENAME_TEMPLATE.format(temp)
    filtered.to_csv(output_path, index=False)
    print(f"Saved {len(filtered)} rows to {output_path}")
