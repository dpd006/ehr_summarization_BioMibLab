import pandas as pd
import os

# 1. Setup Path
base_path = os.path.join('.', 'mimic-iv-clinical-database-demo-2.2', 'hosp')

print("Loading clinical data...")
df_adm = pd.read_csv(os.path.join(base_path, 'admissions.csv.gz'), compression='gzip')
df_diag = pd.read_csv(os.path.join(base_path, 'diagnoses_icd.csv.gz'), compression='gzip')
df_dict = pd.read_csv(os.path.join(base_path, 'd_icd_diagnoses.csv.gz'), compression='gzip')
df_rx = pd.read_csv(os.path.join(base_path, 'prescriptions.csv.gz'), compression='gzip')

# 2. Merge Code Names (So we have text, not numbers)
#    (We use 'inner' join to drop codes we don't have names for)
df_diag = pd.merge(df_diag, df_dict, on=['icd_code', 'icd_version'], how='inner')

# 3. Define the Summarization Function
def summarize_patient(patient_id):
    # Filter for this patient
    admissions = df_adm[df_adm['subject_id'] == patient_id].sort_values('admittime')
    
    if admissions.empty:
        return f"No records found for Patient {patient_id}."

    summary_text = [f"### PATIENT SUMMARY: {patient_id} ###\n"]
    summary_text.append(f"Patient {patient_id} had {len(admissions)} hospital admission(s).\n")

    # Loop through every admission to build the story
    for i, (_, adm) in enumerate(admissions.iterrows(), 1):
        hadm_id = adm['hadm_id']
        date = adm['admittime'].split(' ')[0] # Keep just the YYYY-MM-DD
        
        # Get Diagnoses for this specific visit
        visit_diags = df_diag[df_diag['hadm_id'] == hadm_id]['long_title'].unique()
        # Get Meds for this specific visit
        visit_meds = df_rx[df_rx['hadm_id'] == hadm_id]['drug'].unique()
        
        # Draft the paragraph
        paragraph = (
            f"ADMISSION {i} ({date}):\n"
            f"The patient was admitted to the hospital. "
            f"Primary clinical findings included {', '.join(visit_diags[:5])}"
        )
        
        if len(visit_diags) > 5:
            paragraph += f", among others."
        else:
            paragraph += "."
            
        if len(visit_meds) > 0:
            paragraph += f"\nTreatment included prescriptions for {', '.join(visit_meds[:5])}"
            if len(visit_meds) > 5:
                paragraph += " and other medications."
        else:
            paragraph += "\nNo prescriptions were recorded for this visit."
            
        summary_text.append(paragraph + "\n")

    return "\n".join(summary_text)

# 4. Generate Summaries for a few distinct patients
#    Let's pick 3 random patients from the admissions file
test_patients = df_adm['subject_id'].unique()[:3]

print("-" * 60)
for pid in test_patients:
    print(summarize_patient(pid))
    print("-" * 60)

# Optional: Save one to a file
with open("patient_summary_output.txt", "w") as f:
    f.write(summarize_patient(test_patients[0]))
    print("Saved first summary to 'patient_summary_output.txt'")