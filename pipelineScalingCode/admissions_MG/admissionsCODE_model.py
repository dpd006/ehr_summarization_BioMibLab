import os
import re
import sqlite3

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import json


# ======================
# MODEL CONFIG
# ======================

# Pick the Hugging Face model to use for both SQL generation and clinical summary generation.
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
 
# Other good options:
# "google/gemma-2b-it"
# "google/gemma-7b-it"
# "meta-llama/Llama-3.2-3B-Instruct"
# "mistralai/Mistral-7B-Instruct-v0.2"
# "microsoft/Phi-3-mini-4k-instruct"
# "meta-llama/Llama-3.3-70B-Instruct"

print(f"Loading model: {MODEL_NAME}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,   
    device_map="auto"
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=500,
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
    match = re.search(r"```sql\s*(.*?)```", text, re.IGNORECASE | re.DOTALL)
    if match:
        sql = match.group(1).strip()
    else:
        sql = text.strip()

    sql = sql.replace("```", "").strip()

    if ";" in sql:
        sql = sql.split(";")[0].strip()

    return sql + ";"


def fallback_sql(subject_id):
    return f"""
SELECT
    subject_id,
    hadm_id,
    admittime,
    dischtime,
    deathtime,
    admission_type,
    admitprovider_id,
    admission_location,
    discharge_location,
    insurance,
    language,
    marital_status,
    race,
    edregtime,
    edouttime,
    hospital_expire_flag
FROM admissions
WHERE subject_id = {subject_id}
ORDER BY admittime ASC;
""".strip()


# ======================
# STEP 1: SQL GENERATION
# ======================

def generate_sql(subject_id):
    prompt = f"""
You are a clinical SQL assistant.

Write ONE SQLite query for table admissions with columns:
subject_id, hadm_id, admittime, dischtime, deathtime,
admission_type, admitprovider_id, admission_location, discharge_location,
insurance, language, marital_status, race,
edregtime, edouttime, hospital_expire_flag

Task:
Get all admissions for subject_id = {subject_id}

Rules:
- SELECT only
- FROM admissions
- WHERE subject_id = {subject_id}
- ORDER BY admittime ASC
- return SQL only
- do not return Python
- do not explain anything
""".strip()

    result = generator(prompt, max_new_tokens=500, do_sample=False)
    raw_sql = result[0]["generated_text"][len(prompt):].strip()
    sql = extract_sql(raw_sql)

    if "select" not in sql.lower():
        sql = fallback_sql(subject_id)

    return prompt, raw_sql, sql


# ======================
# STEP 2: BUILD CONTEXT
# ======================
"""
def build_patient_context(subject_id, group):
    lines = [f"Patient ID: {subject_id}"]

    for _, row in group.iterrows():
        lines.append("\nAdmission Event:")

        hadm_id = safe_str(row.get("hadm_id"))
        admittime = safe_str(row.get("admittime"))
        dischtime = safe_str(row.get("dischtime"))
        deathtime = safe_str(row.get("deathtime"))
        admission_type = safe_str(row.get("admission_type"))
        provider = safe_str(row.get("admitprovider_id") or row.get("admitting_provider_id"))
        admission_location = safe_str(row.get("admission_location"))
        discharge_location = safe_str(row.get("discharge_location"))
        insurance = safe_str(row.get("insurance"))
        language = safe_str(row.get("language"))
        marital_status = safe_str(row.get("marital_status"))
        race = safe_str(row.get("race"))
        edregtime = safe_str(row.get("edregtime"))
        edouttime = safe_str(row.get("edouttime"))
        hospital_expire_flag = safe_str(row.get("hospital_expire_flag"))

        if hadm_id:
            lines.append(f"Admission ID: {hadm_id}")
        if admittime:
            lines.append(f"Admitted: {admittime}")
        if dischtime:
            lines.append(f"Discharged: {dischtime}")
        if deathtime:
            lines.append(f"Death time: {deathtime}")
        if admission_type:
            lines.append(f"Admission type: {admission_type}")
        if provider:
            lines.append(f"Admitting provider: {provider}")
        if admission_location:
            lines.append(f"Admission location: {admission_location}")
        if discharge_location:
            lines.append(f"Discharge location: {discharge_location}")
        if insurance:
            lines.append(f"Insurance: {insurance}")
        if language:
            lines.append(f"Language: {language}")
        if marital_status:
            lines.append(f"Marital status: {marital_status}")
        if race:
            lines.append(f"Race: {race}")
        if edregtime:
            lines.append(f"ED registration time: {edregtime}")
        if edouttime:
            lines.append(f"ED out time: {edouttime}")
        if hospital_expire_flag:
            lines.append(f"Hospital expire flag: {hospital_expire_flag}")

    return "\n".join(lines)
"""

def build_patient_context(subject_id, group):
    admissions = []
    for _, row in group.iterrows():
        admissions.append(
            {
                "hadm_id": safe_str(row.get("hadm_id")),
                "admittime": safe_str(row.get("admittime")),
                "dischtime": safe_str(row.get("dischtime")),
                "deathtime": safe_str(row.get("deathtime")),
                "admission_type": safe_str(row.get("admission_type")),
                "admitprovider_id": safe_str(row.get("admitprovider_id") or row.get("admitting_provider_id")),
                "admission_location": safe_str(row.get("admission_location")),
                "discharge_location": safe_str(row.get("discharge_location")),
                "insurance": safe_str(row.get("insurance")),
                "language": safe_str(row.get("language")),
                "marital_status": safe_str(row.get("marital_status")),
                "race": safe_str(row.get("race")),
                "edregtime": safe_str(row.get("edregtime")),
                "edouttime": safe_str(row.get("edouttime")),
                "hospital_expire_flag": safe_str(row.get("hospital_expire_flag")),
            }
        )

    payload = {"patient_id": int(subject_id), "admissions": admissions}
    return json.dumps(payload, ensure_ascii=True)
# ======================
# STEP 3: SUMMARY GENERATION
# ======================

def generate_summary(context):
    prompt = f"""
You are a clinical documentation assistant.

Write ONE short paragraph per admission in chronological order, separated by a blank line.

Hard constraints:
- Paragraphs only: no bullet points, no numbering, no headings.
- Do not copy the input structure or repeat field labels.
- Use ONLY facts present in ADMISSIONS_JSON. Do not invent admission IDs, dates, locations, or events.
- Each paragraph must include the admission ID (hadm_id) and admit/discharge times exactly as given.
- If a field is null/missing, omit it (do not guess).
- Do not mention SQL.
- Do not hallucinate.

ADMISSIONS_JSON:
{context}
""".strip()

    result = generator(prompt, max_new_tokens=500, do_sample=False)
    summary = result[0]["generated_text"][len(prompt):].strip()
    return prompt, summary


# ======================
# MAIN
# ======================

#file_path = "admissions.csv"
#output_file = "admissions_prose.txt"
#sql_file = "admissions_queries.sql"

file_path = "/storage/ice1/0/2/sfatima7/admissions.csv"
output_file = "/storage/ice1/0/2/sfatima7/admissions_prose.txt"
sql_file = "/storage/ice1/0/2/sfatima7/admissions_queries.sql"

df = pd.read_csv(file_path)

print(f"Loaded {len(df)} rows")

conn = sqlite3.connect(":memory:")
df.to_sql("admissions", conn, index=False, if_exists="replace")

subject_ids = df["subject_id"].dropna().astype(int).unique()
print(f"Processing {len(subject_ids)} patients")

with open(output_file, "w", encoding="utf-8") as prose_handle, \
     open(sql_file, "w", encoding="utf-8") as sql_handle:

    for i, subject_id in enumerate(subject_ids):
        try:
            sql_prompt, raw_sql, sql = generate_sql(subject_id)

            sql_handle.write(f"-- Patient {subject_id}\n{sql}\n\n")

            result_df = pd.read_sql_query(sql, conn)
            context = build_patient_context(subject_id, result_df)
            summary_prompt, summary = generate_summary(context)

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
