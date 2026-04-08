import pandas as pd
import os
from datetime import datetime

# 1. Setup Path
base_path = os.path.join('.', 'mimic-iv-clinical-database-demo-2.2', 'hosp')

print("Loading clinical data...")
df_adm = pd.read_csv(os.path.join(base_path, 'admissions.csv.gz'), compression='gzip')
df_diag = pd.read_csv(os.path.join(base_path, 'diagnoses_icd.csv.gz'), compression='gzip')
df_dict = pd.read_csv(os.path.join(base_path, 'd_icd_diagnoses.csv.gz'), compression='gzip')

# Merge Code Names
print("Translating ICD codes...")
df_diag = pd.merge(df_diag, df_dict, on=['icd_code', 'icd_version'], how='inner')

# Ensure dates are actual datetime objects so we can calculate gaps
df_adm['admittime'] = pd.to_datetime(df_adm['admittime'])

# 2. Define the Story Function
def generate_longitudinal_story(patient_id):
    # Get admissions sorted by date
    admissions = df_adm[df_adm['subject_id'] == patient_id].sort_values('admittime')
    
    if admissions.empty:
        return None

    # --- Header ---
    start_date = admissions.iloc[0]['admittime'].strftime('%Y-%m-%d')
    end_date = admissions.iloc[-1]['admittime'].strftime('%Y-%m-%d')
    total_visits = len(admissions)
    
    story = [f"### PATIENT {patient_id} LONGITUDINAL HISTORY ###"]
    story.append(f"Patient tracked from {start_date} to {end_date} ({total_visits} total visits).\n")

    # --- Loop through timeline ---
    prev_discharge = None
    
    for i, (_, adm) in enumerate(admissions.iterrows(), 1):
        hadm_id = adm['hadm_id']
        curr_admit = adm['admittime']
        curr_disch = pd.to_datetime(adm['dischtime'])
        
        # Get Diagnoses for this visit (limit to top 5 distinct important ones)
        diags = list(set(df_diag[df_diag['hadm_id'] == hadm_id]['long_title']))
        diag_str = ", ".join(diags[:5]) if diags else "unknown conditions"
        
        # --- WRITE THE NARRATIVE ---
        date_str = curr_admit.strftime('%Y-%m-%d')
        
        if i == 1:
            # First Visit
            paragraph = (
                f"INITIAL PRESENTATION ({date_str}):\n"
                f"The patient was first admitted for {diag_str}."
            )
        else:
            # Subsequent Visits - Calculate the Gap
            gap = (curr_admit - prev_discharge).days
            paragraph = (
                f"READMISSION ({date_str}, {gap} days later):\n"
                f"The patient returned to the hospital presenting with {diag_str}."
            )

        story.append(paragraph)
        prev_discharge = curr_disch

    story.append("\n" + ("=" * 60) + "\n")
    return "\n".join(story)

# 3. Run for ALL Patients
all_patients = df_adm['subject_id'].unique()
output_filename = "longitudinal_patient_stories.txt"

print(f"Generating stories for {len(all_patients)} patients...")

with open(output_filename, "w", encoding="utf-8") as f:
    for pid in all_patients:
        summary = generate_longitudinal_story(pid)
        if summary:
            f.write(summary)

print(f"[DONE] Saved stories to: {output_filename}")