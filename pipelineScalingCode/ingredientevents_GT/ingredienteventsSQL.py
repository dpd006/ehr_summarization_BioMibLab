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

def clean_status(value):
    if not isinstance(value, str):
        return None
    
    v = value.strip().lower()

    if v == "changedose/rate":
        return "infusion rate was changed"
    if v == "finishedrunning":
        return "finished running"
    if v == "stopped":
        return "was stopped"
    if v == "paused":
        return "was paused"

    return v

def fmt_num(x, digits=2):
    if x is None or pd.isna(x):
        return None
    try:
        return f"{float(x):.{digits}f}"
    except:
        return str(x)

#Converting a row from icustays to descriptive prose
def row_to_prose(row):
    sentences = []

    subject_id = safe_str(row.get("subject_id"))
    hadm_id = safe_str(row.get("hadm_id"))
    stay_id = safe_str(row.get("stay_id"))

    starttime = safe_str(row.get("starttime"))
    endtime = safe_str(row.get("endtime"))
    drug_label = safe_str(row.get("label"))   # ← from d_items
    
    # patient id and timing
    if subject_id and starttime and endtime:
        sentences.append(f"Patient {subject_id} received {drug_label or 'medication'} from {starttime} to {endtime}.")
    # IDs
    if hadm_id:
        sentences.append(f"The hospital admission ID was {hadm_id}.")
    if stay_id:
        sentences.append(f"The ICU stay ID was {stay_id}.")
    
    # Ingredient info
    itemid = safe_str(row.get("itemid"))
    

    amount = fmt_num(row.get("amount"))
    amountuom = safe_str(row.get("amountuom"))

    rate = fmt_num(row.get("rate"))
    rateuom = safe_str(row.get("rateuom"))

    status = clean_status(row.get("statusdescription"))

    originalrate = fmt_num(row.get("originalrate"))

    #if itemid:
    #    sentences.append(f"The medication item ID was {itemid}.")
    if amount and amountuom:
        sentences.append(f"The recorded amount was {amount} {amountuom}.")
    if rate and rateuom:
        sentences.append(f"The infusion rate was {rate} {rateuom}.")
    

    # ONLY include original rate if the status indicates a dose/rate change
    if status and ("rate" in status) and originalrate:
        sentences.append(f"The infusion rate was changed from {originalrate} {rateuom}.")
    elif status:
        sentences.append(f"The infusion {status}.")
    
    return " ".join(sentences)

#------MAIN--------    

# CSV path
ingredient_path = os.path.join("mimic-iv-demo", "mimic-iv-clinical-database-demo-2.2", "icu", "ingredientevents.csv")
items_path = os.path.join("mimic-iv-demo", "mimic-iv-clinical-database-demo-2.2", "icu", "d_items.csv")

# Load CSV
df = pd.read_csv(ingredient_path)
print(f"Loaded ingredientevents: {df.shape}")

df_items = pd.read_csv(items_path)
df = df.merge(df_items[["itemid", "label"]], on="itemid", how="left")

# Convert to prose
prose_descriptions = df.apply(row_to_prose, axis=1).tolist()

# Save to a single text file
output_file = r"pipelineScalingCode/ingredientevents_prose.txt"
with open(output_file, "w", encoding="utf-8") as f:
    for line in prose_descriptions:
        f.write(line + "\n\n")  # double newline between patients

print(f"Saved {len(prose_descriptions)} patient descriptions to {output_file}")
    
# Creating ingredientevents_queries.sql file

subject_ids = df["subject_id"].dropna().unique()
print(f"Found {len(subject_ids)} ICU patients")
output_sql = r"pipelineScalingCode/ingredientevents_queries.sql"

with open(output_sql, "w", encoding="utf-8") as f:
    for pid in subject_ids:
        f.write(f"""-- subject_id: {pid}
SELECT *
FROM ingredientevents
WHERE subject_id = {pid}
ORDER BY starttime DESC;

""")
print(f"[DONE] Wrote ICU queries to {output_sql}")
    