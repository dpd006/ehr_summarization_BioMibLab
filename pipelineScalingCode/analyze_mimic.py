import pandas as pd
import os

# 1. Setup path
base_path = os.path.join('.', 'mimic-iv-clinical-database-demo-2.2', 'hosp')

# 2. Load necessary files
print("Loading data...")
df_diag = pd.read_csv(os.path.join(base_path, 'diagnoses_icd.csv.gz'), compression='gzip')
df_dict = pd.read_csv(os.path.join(base_path, 'd_icd_diagnoses.csv.gz'), compression='gzip')
df_rx = pd.read_csv(os.path.join(base_path, 'prescriptions.csv.gz'), compression='gzip')
# We also need admissions to get the dates
df_adm = pd.read_csv(os.path.join(base_path, 'admissions.csv.gz'), compression='gzip')

# 3. Merge Diagnosis Names
df_diag = pd.merge(df_diag, df_dict, on=['icd_code', 'icd_version'], how='left')

# 4. Filter for our specific patient
patient_id = 10000032
patient_adm = df_adm[df_adm['subject_id'] == patient_id].sort_values('admittime')

print(f"\n=== Timeline for Patient {patient_id} ===\n")

for _, adm in patient_adm.iterrows():
    hadm_id = adm['hadm_id']
    admit_date = adm['admittime']
    disch_date = adm['dischtime']
    
    print(f"Admission ID: {hadm_id}")
    print(f"   Date: {admit_date} to {disch_date}")
    
    # Get Diagnoses for THIS admission
    current_diag = df_diag[df_diag['hadm_id'] == hadm_id]
    print(f"   Diagnoses ({len(current_diag)}):")
    for _, row in current_diag.iterrows():
        print(f"     - {row['long_title']}")
        
    # Get Meds for THIS admission
    current_rx = df_rx[df_rx['hadm_id'] == hadm_id]['drug'].unique()
    print(f"   Meds ({len(current_rx)}):")
    # Print only first 5 to save space
    for drug in current_rx[:5]:
        print(f"     - {drug}")
    if len(current_rx) > 5:
        print(f"     - ... and {len(current_rx)-5} more.")
        
    print("-" * 50)