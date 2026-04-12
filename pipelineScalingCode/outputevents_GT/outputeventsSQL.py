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

def row_to_prose(row):
    sentences = []

    subject_id = safe_str(row.get("subject_id"))
    hadm_id = safe_str(row.get("hadm_id"))
    stay_id = safe_str(row.get("stay_id"))

    charttime = safe_str(row.get("charttime"))
    itemid = safe_str(row.get("itemid"))
    output_label = safe_str(row.get("label"))   # ← from d_items

    amount = safe_str(row.get("value"))
    amountuom = safe_str(row.get("valueuom"))

    # Patient id, event, and timing
    if subject_id and charttime:
        if output_label:
            sentences.append(f"Patient {subject_id} had output from {output_label} at {charttime}.")
        else:
            sentences.append(f"Patient {subject_id} had an output at {charttime}.")
    
    # IDs
    if hadm_id:
        sentences.append(f"The hospital admission ID was {hadm_id}.")
    if stay_id:
        sentences.append(f"The ICU stay ID was {stay_id}.")
    
    # Amount
    if amount and amountuom:
        sentences.append(f"The recorded amount was {amount} {amountuom}.")
    
    return " ".join(sentences)

#------MAIN--------

# CSV path
outputevents_path = os.path.join("mimic-iv-demo", "mimic-iv-clinical-database-demo-2.2", "icu", "outputevents.csv")
items_path = os.path.join("mimic-iv-demo", "mimic-iv-clinical-database-demo-2.2", "icu", "d_items.csv")

# Load CSV
df = pd.read_csv(outputevents_path)
print(f"Loaded outputevents: {df.shape}")

df_items = pd.read_csv(items_path)
df = df.merge(df_items[["itemid", "label"]], on="itemid", how="left")

# Convert to prose
prose_descriptions = df.apply(row_to_prose, axis=1).tolist()

# Save to a single text file
output_file = r"pipelineScalingCode/outputevents_prose.txt"
with open(output_file, "w", encoding="utf-8") as f:
    for line in prose_descriptions:
        f.write(line + "\n\n")  # double newline between patients

print(f"Saved {len(prose_descriptions)} patient descriptions to {output_file}")


