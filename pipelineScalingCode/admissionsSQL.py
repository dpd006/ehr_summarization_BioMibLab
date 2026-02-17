import pandas as pd

def safe_lower(value):
    """Return lowercase string if possible, else None."""
    if isinstance(value, str) and value.strip():
        return value.lower()
    return None

def safe_str(value):
    """Return string if not NaN, else None."""
    if pd.isna(value) or str(value).strip() == '':
        return None
    return str(value)

def row_to_prose(row):
    """Convert a row from admissions CSV to descriptive prose."""
    sentences = []

    # Admission/discharge times
    subject_id = safe_str(row.get('subject_id'))
    adm_time = safe_str(row.get('admittime'))
    dis_time = safe_str(row.get('dischtime'))
    hadm_id = safe_str(row.get('hadm_id'))
    if subject_id and adm_time and dis_time:
        sentences.append(f"Patient {subject_id} was admitted on {adm_time} and discharged on {dis_time}.")
    if hadm_id:
        sentences.append(f"Their hospital admission ID was {hadm_id}.")

    # Death info
    deathtime = safe_str(row.get('deathtime'))
    if deathtime:
        sentences.append(f"They died on {deathtime}.")

    # Admission type and provider
    admission_type = safe_lower(row.get('admission_type'))
    provider = safe_str(row.get('admitprovider_id') or row.get('admitting_provider_id'))
    if admission_type:
        if provider:
            sentences.append(f"The admission type was {admission_type}, and they were admitted by provider {provider}.")
        else:
            sentences.append(f"The admission type was {admission_type}.")

    # Admission/discharge location
    admission_loc = safe_lower(row.get('admission_location'))
    discharge_loc = safe_lower(row.get('discharge_location'))
    if admission_loc or discharge_loc:
        sentences.append(f"They arrived via {admission_loc or 'unknown'} and were discharged to {discharge_loc or 'unknown'}.")

    # Insurance, language, marital status
    insurance = safe_str(row.get('insurance'))
    language = safe_str(row.get('language'))
    marital = safe_lower(row.get('marital_status'))
    parts = []
    if insurance: parts.append(f"Insurance: {insurance}")
    if language: parts.append(f"Language: {language}")
    if marital: parts.append(f"Marital status: {marital}")
    if parts:
        sentences.append(", ".join(parts) + ".")

    # Race
    race = safe_str(row.get('race'))
    if race:
        sentences.append(f"Race: {race}.")

    # ED times
    edregtime = safe_str(row.get('edregtime'))
    edouttime = safe_str(row.get('edouttime'))
    if edregtime or edouttime:
        sentences.append(f"ED registration: {edregtime or 'unknown'}, ED out: {edouttime or 'unknown'}.")

    # Hospital expire flag
    hospital_flag = safe_str(row.get('hospital_expire_flag'))
    if hospital_flag:
        sentences.append(f"Hospital expire flag: {hospital_flag}.")

    return " ".join(sentences)

# --- MAIN SCRIPT ---

# CSV path
file_path = r"C:\Users\prana\OneDrive - Georgia Institute of Technology\ResearchMIBLAB\ehr_summarization_BioMibLab\mimic-iv-demo\mimic-iv-clinical-database-demo-2.2\hosp\admissions\admissions.csv"

# Load CSV
df = pd.read_csv(file_path)
print(f"Loaded admissions: {df.shape}")

# Convert to prose
prose_descriptions = df.apply(row_to_prose, axis=1).tolist()

# Save to a single text file
output_file = r"C:\Users\prana\OneDrive - Georgia Institute of Technology\ResearchMIBLAB\ehr_summarization_BioMibLab\pipelineScalingCode\output\admissions_prose.txt"
with open(output_file, "w", encoding="utf-8") as f:
    for line in prose_descriptions:
        f.write(line + "\n\n")  # double newline between patients

print(f"Saved {len(prose_descriptions)} patient descriptions to {output_file}")
