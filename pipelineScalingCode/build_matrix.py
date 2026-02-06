import pandas as pd
import os

# 1. Setup Path
base_path = os.path.join('.', 'mimic-iv-clinical-database-demo-2.2', 'hosp')

print("Loading data...")
# Load Diagnoses and Prescriptions
df_diag = pd.read_csv(os.path.join(base_path, 'diagnoses_icd.csv.gz'), compression='gzip')
df_rx = pd.read_csv(os.path.join(base_path, 'prescriptions.csv.gz'), compression='gzip')

# ---------------------------------------------------------
# 2. PRE-PROCESSING
# ---------------------------------------------------------
# For the matrix, we only care about: hadm_id (Row), icd_code (Column), and drug (Column)

# A. Clean Diagnoses
# Let's keep only the top 20 most common codes to keep the terminal output readable
# (In real research, you might keep hundreds or group them)
top_codes = df_diag['icd_code'].value_counts().head(20).index
df_diag_filtered = df_diag[df_diag['icd_code'].isin(top_codes)].copy()
df_diag_filtered['value'] = 1 # Marker for pivot

# B. Clean Prescriptions
# Keep top 20 drugs
top_drugs = df_rx['drug'].value_counts().head(20).index
df_rx_filtered = df_rx[df_rx['drug'].isin(top_drugs)].copy()
df_rx_filtered['value'] = 1

# ---------------------------------------------------------
# 3. PIVOTING (Creating the 0/1 Matrix)
# ---------------------------------------------------------
print("Building Matrix...")

# Pivot Diagnoses: Rows=Admission, Cols=ICD Code
diag_matrix = df_diag_filtered.pivot_table(
    index='hadm_id', 
    columns='icd_code', 
    values='value', 
    fill_value=0,
    aggfunc='max' # Ensure we don't get 2s if recorded twice
)
# Add prefix so we know these are diagnoses
diag_matrix.columns = ['Dx_' + str(col) for col in diag_matrix.columns]

# Pivot Meds: Rows=Admission, Cols=Drug Name
rx_matrix = df_rx_filtered.pivot_table(
    index='hadm_id', 
    columns='drug', 
    values='value', 
    fill_value=0,
    aggfunc='max'
)
# Add prefix
rx_matrix.columns = ['Rx_' + str(col) for col in rx_matrix.columns]

# ---------------------------------------------------------
# 4. MERGE EVERYTHING
# ---------------------------------------------------------
# Join them on Admission ID (hadm_id)
final_matrix = pd.merge(diag_matrix, rx_matrix, on='hadm_id', how='outer').fillna(0).astype(int)

print(f"\nMatrix Shape: {final_matrix.shape}")
print(f"(Rows = Hospital Admissions, Columns = Features)")
print("-" * 30)
print("First 5 rows of your Machine Learning Matrix:")
print("-" * 30)

# Print a subset of columns to make it fit on screen
print(final_matrix.iloc[:5, :5]) 

# Optional: Save to CSV for your actual research
# final_matrix.to_csv("processed_features.csv")
# print("\nSaved to processed_features.csv")