import pandas as pd
import os

# 1. Setup (Same as before)
base_path = os.path.join('.', 'mimic-iv-clinical-database-demo-2.2', 'hosp')

# Load the dictionary and the matrix we just built
# (Note: In a real workflow, you would pass the matrix from the previous script, 
#  but here we will quickly rebuild it to be self-contained)

print("Rebuilding matrix to rename columns...")
df_diag = pd.read_csv(os.path.join(base_path, 'diagnoses_icd.csv.gz'), compression='gzip')
df_dict = pd.read_csv(os.path.join(base_path, 'd_icd_diagnoses.csv.gz'), compression='gzip')

# Filter top 20 codes again
top_codes = df_diag['icd_code'].value_counts().head(20).index
df_diag_filtered = df_diag[df_diag['icd_code'].isin(top_codes)].copy()
df_diag_filtered['value'] = 1

# Pivot
matrix = df_diag_filtered.pivot_table(index='hadm_id', columns='icd_code', values='value', fill_value=0, aggfunc='max')

# ---------------------------------------------------------
# RENAME COLUMNS (The New Part)
# ---------------------------------------------------------
# Create a lookup dictionary: {'4019': 'Hypertension', ...}
code_map = dict(zip(df_dict['icd_code'], df_dict['long_title']))

# Rename the columns
new_columns = []
for col in matrix.columns:
    # Get the name, truncate it to 20 chars so it fits on screen
    name = code_map.get(col, "Unknown")[:20] 
    new_columns.append(f"Dx_{name}")

matrix.columns = new_columns

print("-" * 50)
print("Readable Matrix (First 5 rows):")
print("-" * 50)
print(matrix.iloc[:5, :5])

# Save this for your research
matrix.to_csv("human_readable_dataset.csv")
print("\n[Done] Saved as 'human_readable_dataset.csv'")