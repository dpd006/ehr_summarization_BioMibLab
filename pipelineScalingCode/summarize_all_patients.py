import pandas as pd
import os

# 1. Setup Path to your data
base_path = os.path.join('.', 'mimic-iv-clinical-database-demo-2.2', 'hosp')

print("Loading clinical data...")
# Load the 4 key files
df_adm = pd.read_csv(os.path.join(base_path, 'admissions.csv.gz'), compression='gzip')
df_diag = pd.read_csv(os.path.join(base_path, 'diagnoses_icd.csv.gz'), compression='gzip')
df_dict = pd.read_csv(os.path.join(base_path, 'd_icd_diagnoses.csv.gz'), compression='gzip')
df_rx = pd.read_csv(os.path.join(base_path, 'prescriptions.csv.gz'), compression='gzip')

# 2. Merge Code Names (Translate "4019" -> "Hypertension")
print("Translating ICD codes...")
df_diag = pd.merge(df_diag, df_dict, on=['icd_code', 'icd_version'], how='inner')

# 3. Define the Summary Function
def get_patient_summary(patient_id):
    # Get all admissions for this patient, sorted by date
    admissions = df_adm[df_adm['subject_id'] == patient_id].sort_values('admittime')
    
    if admissions.empty:
        return None

    # Start the text block for this patient
    text_block = [f"=== PATIENT {patient_id} HISTORY ({len(admissions)} visits) ==="]

    for i, (_, adm) in enumerate(admissions.iterrows(), 1):
        hadm_id = adm['hadm_id']
        date_str = adm['admittime'].split(' ')[0] # YYYY-MM-DD
        
        # Get data for THIS specific visit
        # We use sets to remove exact duplicates, then convert to list
        diags = list(set(df_diag[df_diag['hadm_id'] == hadm_id]['long_title']))
        meds = list(set(df_rx[df_rx['hadm_id'] == hadm_id]['drug']))
        
        # Format the paragraph
        # (We limit lists to 10 items so the text is readable, but you can increase this)
        diag_text = ", ".join(diags[:10]) if diags else "No recorded diagnoses"
        if len(diags) > 10: diag_text += "..."
        
        med_text = ", ".join(meds[:10]) if meds else "No recorded medications"
        if len(meds) > 10: med_text += "..."

        visit_summary = (
            f"\n[Visit {i} - {date_str}]\n"
            f"  * DIAGNOSES: {diag_text}\n"
            f"  * MEDICATIONS: {med_text}"
        )
        text_block.append(visit_summary)

    text_block.append("\n" + ("-" * 60) + "\n") # Separator between patients
    return "\n".join(text_block)

# 4. Run for ALL Patients
all_patients = df_adm['subject_id'].unique()
total = len(all_patients)
output_filename = "full_mimic_summaries.txt"

print(f"Generating summaries for {total} patients...")

with open(output_filename, "w", encoding="utf-8") as f:
    for i, pid in enumerate(all_patients, 1):
        summary = get_patient_summary(pid)
        if summary:
            f.write(summary)
        
        # Show progress every 10 patients
        if i % 10 == 0:
            print(f"Processed {i}/{total}...")

print(f"\n[DONE] All summaries saved to: {os.path.abspath(output_filename)}")