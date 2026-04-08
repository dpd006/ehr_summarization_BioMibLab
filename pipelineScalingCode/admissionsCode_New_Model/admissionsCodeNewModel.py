import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline


# ======================
# MODEL CONFIG
# ======================

# CHANGE THIS to any model you want
MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"

# Other good options:
# "google/gemma-2b-it"
# "google/gemma-7b-it"
# "meta-llama/Llama-3.2-3B-Instruct"
# "mistralai/Mistral-7B-Instruct-v0.2"

print(f"Loading model: {MODEL_NAME}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,   # CPU safe
    device_map="cpu"
)

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=300,
    temperature=0.3,
    do_sample=True
)


# ======================
# HELPERS
# ======================

def safe_str(value):
    if pd.isna(value) or str(value).strip() == '':
        return None
    return str(value).strip()


def build_patient_context(subject_id, group):
    lines = [f"Patient ID: {subject_id}"]

    for _, row in group.iterrows():

        hadm_id = safe_str(row.get('hadm_id'))
        admittime = safe_str(row.get('admittime'))
        dischtime = safe_str(row.get('dischtime'))
        admission_type = safe_str(row.get('admission_type'))
        insurance = safe_str(row.get('insurance'))
        race = safe_str(row.get('race'))

        lines.append(f"\nAdmission ID: {hadm_id}")

        if admittime:
            lines.append(f"Admitted: {admittime}")

        if dischtime:
            lines.append(f"Discharged: {dischtime}")

        if admission_type:
            lines.append(f"Admission type: {admission_type}")

        if insurance:
            lines.append(f"Insurance: {insurance}")

        if race:
            lines.append(f"Race: {race}")

    return "\n".join(lines)


def generate_summary(context):

    prompt = f"""
You are a clinical documentation assistant.

Summarize this patient's hospital admissions clearly and concisely.

{context}
"""

    result = generator(prompt)

    return result[0]["generated_text"][len(prompt):].strip()


def generate_sql(subject_id):

    return f"""
SELECT *
FROM admissions
WHERE subject_id = {subject_id}
ORDER BY admittime;
"""


# ======================
# MAIN
# ======================

file_path = "admissions.csv"
output_file = "admissions_prose.txt"
sql_file = "admissions_queries.sql"

df = pd.read_csv(file_path)

print(f"Loaded {len(df)} rows")

grouped = df.groupby("subject_id")

with open(output_file, "w") as f:

    for i, (subject_id, group) in enumerate(grouped):

        context = build_patient_context(subject_id, group)

        summary = generate_summary(context)

        f.write(f"=== Patient {subject_id} ===\n")
        f.write(summary + "\n\n")

        print(f"Done {i+1}/{len(grouped)}")


with open(sql_file, "w") as f:

    for subject_id in df["subject_id"].unique():

        f.write(generate_sql(subject_id))
        f.write("\n\n")


print("Finished everything.")