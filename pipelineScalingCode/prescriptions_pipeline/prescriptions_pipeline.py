import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import sqlite3
import re


# ======================
# MODEL CONFIG
# ======================

MODEL_NAME = "google/gemma-2b-it"  # lighter + easier to run

print(f"Loading model: {MODEL_NAME}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=200,
    do_sample=False,
    pad_token_id=tokenizer.eos_token_id
)


# ======================
# HELPERS
# ======================

def safe_str(value):
    if pd.isna(value) or str(value).strip() == "":
        return None
    return str(value).strip()


def extract_sql(text):
    match = re.search(r"```sql\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if match:
        sql = match.group(1)
    else:
        sql = text

    sql = sql.replace("```", "").strip()

    if ";" in sql:
        sql = sql.split(";")[0]

    return sql.strip() + ";"


# ======================
# STEP 1: SQL GENERATION
# ======================

def generate_sql(subject_id):
    prompt = f"""
You are a clinical SQL assistant.

Write ONE SQLite query for table prescriptions with columns:
subject_id, hadm_id, drug, starttime, stoptime, route, dose_val_rx, dose_unit_rx, prod_strength

Task:
Get all prescriptions for subject_id = {subject_id}

Rules:
- SELECT only
- FROM prescriptions
- WHERE subject_id = {subject_id}
- ORDER BY starttime ASC
- return SQL only
"""

    result = generator(prompt)
    raw = result[0]["generated_text"][len(prompt):].strip()
    sql = extract_sql(raw)

    if "select" not in sql.lower():
        raise ValueError(f"Invalid SQL generated: {raw}")

    return prompt, raw, sql


# ======================
# STEP 2: BUILD CONTEXT
# ======================

def build_patient_context(subject_id, group):
    lines = [f"Patient ID: {subject_id}"]

    for _, row in group.iterrows():
        drug = safe_str(row.get("drug"))
        hadm_id = safe_str(row.get("hadm_id"))
        start = safe_str(row.get("starttime"))
        stop = safe_str(row.get("stoptime"))
        route = safe_str(row.get("route"))
        dose = safe_str(row.get("dose_val_rx"))
        unit = safe_str(row.get("dose_unit_rx"))
        strength = safe_str(row.get("prod_strength"))

        lines.append("\nMedication Event:")

        if hadm_id:
            lines.append(f"Hospital Admission ID: {hadm_id}")
        if drug:
            lines.append(f"Drug: {drug}")
        if start:
            lines.append(f"Start: {start}")
        if stop:
            lines.append(f"Stop: {stop}")
        if route:
            lines.append(f"Route: {route}")
        if dose:
            lines.append(f"Dose: {dose} {unit if unit else ''}")
        if strength:
            lines.append(f"Strength: {strength}")

    return "\n".join(lines)


# ======================
# STEP 3: SUMMARY GENERATION
# ======================

def generate_summary(context):
    prompt = f"""
You are a clinical pharmacology assistant.

Summarize this patient's medication history clearly.
Focus on:
- drugs used
- timing patterns
- routes of administration

Do NOT hallucinate.

{context}
"""

    result = generator(prompt)
    summary = result[0]["generated_text"][len(prompt):].strip()

    return prompt, summary


# ======================
# MAIN
# ======================

file_path = r"mimic-iv-demo/mimic-iv-clinical-database-demo-2.2/hosp/prescriptions/prescriptions.csv"

output_file = r"pipelineScalingCode/output/prescriptions_prose.txt"
sql_file = r"pipelineScalingCode/output/prescriptions_queries.sql"

df = pd.read_csv(file_path)

print(f"Loaded {len(df)} rows")
print(f"Columns: {list(df.columns)}")

conn = sqlite3.connect(":memory:")
df.to_sql("prescriptions", conn, index=False, if_exists="replace")

subject_ids = df["subject_id"].dropna().astype(int).unique()
print(f"Processing {len(subject_ids)} patients")

os.makedirs(os.path.dirname(output_file), exist_ok=True)


with open(output_file, "w", encoding="utf-8") as prose_handle, \
     open(sql_file, "w", encoding="utf-8") as sql_handle:

    for i, subject_id in enumerate(subject_ids):

        try:
            # STEP 1: SQL
            sql_prompt, raw_sql, sql = generate_sql(subject_id)

            sql_handle.write(f"-- Patient {subject_id}\n{sql}\n\n")

            # STEP 2: EXECUTE SQL
            result_df = pd.read_sql_query(sql, conn)

            # STEP 3: CONTEXT
            context = build_patient_context(subject_id, result_df)

            # STEP 4: SUMMARY
            summary_prompt, summary = generate_summary(context)

            # STEP 5: LOG EVERYTHING
            prose_handle.write(f"=== Patient {subject_id} ===\n")
            prose_handle.write("SQL PROMPT:\n" + sql_prompt + "\n\n")
            prose_handle.write("RAW SQL OUTPUT:\n" + raw_sql + "\n\n")
            prose_handle.write("EXECUTED SQL:\n" + sql + "\n\n")
            prose_handle.write("SUMMARY PROMPT:\n" + summary_prompt + "\n\n")
            prose_handle.write("SUMMARY OUTPUT:\n" + summary + "\n\n")

            print(f"Done {i+1}/{len(subject_ids)}")

        except Exception as e:
            print(f"Failed for patient {subject_id}: {e}")


conn.close()

print("Finished everything.")