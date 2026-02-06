import pandas as pd
import os

# 1. Setup Path
base_path = os.path.join('.', 'mimic-iv-clinical-database-demo-2.2', 'hosp')

print("Loading clinical data (this happens once)...")
df_adm = pd.read_csv(os.path.join(base_path, 'admissions.csv.gz'), compression='gzip')
df_diag = pd.read_csv(os.path.join(base_path, 'diagnoses_icd.csv.gz'), compression='gzip')
df_dict = pd.read_csv(os.path.join(base_path, 'd_icd_diagnoses.csv.gz'), compression='gzip')
df_rx = pd.read_csv(os.path.join(base_path, 'prescriptions.csv.gz'), compression='gzip')

# 2. Merge Code Names
print("Merging diagnosis dictionaries...")
df_diag = pd.merge(df_diag, df_dict, on=['icd_code', 'icd_version'], how='inner')

# 3. Summarization Function
def generate_summary(patient_id):
    # Filter for this patient
    admissions = df_adm[df_adm['subject_id'] == patient_id].sort_values('admittime')
    
    if admissions.empty:
        return ""

    # Start the narrative
    summary_parts = [f"Patient {patient_id} is a {len(admissions)}-visit patient."]

    for i, (_, adm) in enumerate(admissions.iterrows(), 1):
        hadm_id = adm['hadm_id']
        date = adm['admittime'].split(' ')[0]
        
        # Get Diagnoses
        visit_diags = df_diag[df_diag['hadm_id'] == hadm_id]['long_title'].unique()
        # Get Meds
        visit_meds = df_rx[df_rx['hadm_id'] == hadm_id]['drug'].unique()
        
        # Build paragraph
        # We limit to top 8 items to keep the summary concise but informative
        diag_str = ", ".join(visit_diags[:8]) if len(visit_diags) > 0 else "No recorded diagnoses"
        med_str = ", ".join(visit_meds[:8]) if len(visit_meds) > 0 else "No recorded medications"
        
        paragraph = (
            f" Visit {i} ({date}): "
            f"Admitted with {diag_str}. "
            f"Prescribed {med_str}."
        )
        summary_parts.append(paragraph)

    return "\n".join(summary_parts)

# 4. Processing ALL Patients
all_patients = df_adm['subject_id'].unique()
total_patients = len(all_patients)

print(f"\nProcessing {total_patients} patients...")

results = []

# Loop through everyone
for i, pid in enumerate(all_patients, 1):
    summary = generate_summary(pid)
    if summary:
        results.append({'subject_id': pid, 'summary': summary})
    
    # Simple progress bar
    if i % 10 == 0:
        print(f" - Processed {i}/{total_patients} patients...")

# 5. Save Outputs
print("\nSaving files...")

# Option A: Save as CSV (Best for Data Science/ML)
df_results = pd.DataFrame(results)
df_results.to_csv("all_patient_summaries.csv", index=False)
print(" [OK] Saved 'all_patient_summaries.csv' (Load this into Python/Pandas)")

# Option B: Save as Text File (Best for human reading)
with open("all_patient_summaries.txt", "w", encoding='utf-8') as f:
    for item in results:
        f.write("-" * 50 + "\n")
        f.write(f"### ID: {item['subject_id']} ###\n")
        f.write(item['summary'] + "\n\n")

print(" [OK] Saved 'all_patient_summaries.txt' (Open this to read)")