import pandas as pd
import os

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

#Converting a row from icustays to descriptive prose
def row_to_prose(row):

    sentences = []

    subject_id = safe_str(row.get("subject_id"))
    hadm_id = safe_str(row.get("hadm_id"))
    stay_id = safe_str(row.get("stay_id"))
    intime = safe_str(row.get("intime"))
    outtime = safe_str(row.get("outtime"))
    los = safe_str(row.get("los"))

    first_careunit = safe_lower(row.get("first_careunit"))
    last_careunit = safe_lower(row.get("last_careunit"))

    if subject_id and intime and outtime:
        sentences.append(f"Patient {subject_id} was admitted to ICU on {intime} and discharged on {outtime}.")
    if hadm_id:
        sentences.append(f"Their hospital admission ID was {hadm_id}.")
    if stay_id:
        sentences.append(f"The ICU stay ID was {stay_id}.")

    if first_careunit != last_careunit and los:
        sentences.append(
            f"They were admitted to {first_careunit or 'unknown'} "
            f"and moved to {last_careunit or 'unknown'} for total of {los} days."
        )
    elif first_careunit == last_careunit or los:
        sentences.append(f"They were admitted to {first_careunit} for {los} days")
    
    return " ".join(sentences)


# CSV path
file_path = os.path.join("mimic-iv-demo", "mimic-iv-clinical-database-demo-2.2", "icu", "icustays.csv")

# Load CSV
df = pd.read_csv(file_path)
print(f"Loaded icustays: {df.shape}")

# Convert to prose
prose_descriptions = df.apply(row_to_prose, axis=1).tolist()

# Save to a single text file
output_file = r"pipelineScalingCode/icustays_prose.txt"
with open(output_file, "w", encoding="utf-8") as f:
    for line in prose_descriptions:
        f.write(line + "\n\n")  # double newline between patients

print(f"Saved {len(prose_descriptions)} patient descriptions to {output_file}")

# Creating icustays_queries.sql file
subject_ids = df["subject_id"].dropna().unique()
print(f"Found {len(subject_ids)} ICU patients")
output_sql = r"pipelineScalingCode/icustays_queries.sql"

with open(output_sql, "w", encoding="utf-8") as f:
    for pid in subject_ids:
        f.write(f"""-- subject_id: {pid}
SELECT *
FROM icustays
WHERE subject_id = {pid}
ORDER BY intime DESC;

""")
print(f"[DONE] Wrote ICU queries to {output_sql}")

