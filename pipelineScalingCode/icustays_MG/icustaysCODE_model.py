import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import sqlite3
import re
 
 
# ======================
# MODEL CONFIG
# ======================
 
# Pick the Hugging Face model to use for both SQL generation and clinical summary generation.
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")

# Other good options:
# "google/gemma-2b-it"
# "google/gemma-7b-it"
# "meta-llama/Llama-3.2-3B-Instruct"
# "mistralai/Mistral-7B-Instruct-v0.2"
# "microsoft/Phi-3-mini-4k-instruct"
# "meta-llama/Llama-3.3-70B-Instruct"
 
print(f"Loading model: {MODEL_NAME}")

# Load the tokenizer so text can be converted into model tokens.
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, token=HF_TOKEN)
 
# Load the actual language model on CPU.
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    token=HF_TOKEN,
    dtype=torch.float16,   
    device_map="auto"
)

# Some models do not define a pad token, so we reuse the EOS token.
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Create a text-generation pipeline so we can call the model more easily.
generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=400,
    do_sample=False,
    pad_token_id=tokenizer.eos_token_id

)
 
 # ======================
# HELPERS
# ======================
 
# Convert missing/blank values into None so they do not appear as messy strings like "nan" in the final patient context.
def safe_str(value):
    if pd.isna(value) or str(value).strip() == '':
        return None
    return str(value).strip()
 
# Clean up the model's SQL response so it can actually run in SQLite. This removes ```sql fences and keeps just one executable query.
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

# Ask the LLM to generate SQL for one patient.
# Return:
# 1. the prompt sent to the model
# 2. the raw model output
# 3. the cleaned SQL that will be executed
def generate_sql(subject_id):
    prompt = f"""
You are a clinical SQL assistant.

Write one SQLite query for the table icustays with these columns:
subject_id, hadm_id, stay_id, first_careunit, last_careunit, intime, outtime, los

Task:
Get all ICU stays for subject_id = {subject_id}

Requirements:
- Use SELECT only
- Query only from icustays
- Filter WHERE subject_id = {subject_id}
- Order by intime ASC
- Return SQL only
- Do not return Python
- Do not explain anything
- Do not use markdown unless it is a sql code block
"""

    result = generator(prompt)
    raw_sql_output = result[0]["generated_text"][len(prompt):].strip()
    sql = extract_sql(result[0]["generated_text"][len(prompt):].strip())
    if not sql.lower().strip().startswith("select"):
        raise ValueError(f"Model did not return SQL: {raw_sql_output}")

    return prompt, raw_sql_output, sql

# Turn the SQL query results into readable patient context text.
# This text is what gets sent into the second LLM step for summarization.
def build_patient_context(subject_id, group):
    lines = [f"Patient ID: {subject_id}"]

    for _, row in group.iterrows():
        hadm_id = safe_str(row.get("hadm_id"))
        stay_id = safe_str(row.get("stay_id"))
        first_careunit = safe_str(row.get("first_careunit"))
        last_careunit = safe_str(row.get("last_careunit"))
        intime = safe_str(row.get("intime"))
        outtime = safe_str(row.get("outtime"))
        los = safe_str(row.get("los"))

        lines.append(f"\nHospital Admission ID: {hadm_id}")
        lines.append(f"ICU Stay ID: {stay_id}")

        if first_careunit:
            lines.append(f"First care unit: {first_careunit}")

        if last_careunit:
            lines.append(f"Last care unit: {last_careunit}")

        if intime:
            lines.append(f"ICU admitted: {intime}")

        if outtime:
            lines.append(f"ICU discharged: {outtime}")

        if los:
            lines.append(f"Length of stay (days): {los}")

    return "\n".join(lines)

# Ask the LLM to convert the structured ICU context into a short narrative summary.
# Return both the prompt and the final generated summary.
def generate_summary(context):
    prompt = f"""
You are a clinical documentation assistant.

Summarize this patient's ICU stays clearly and concisely.
Focus on ICU timing, care units, and length of stay.
Do not mention SQL.

{context}
"""

    result = generator(prompt)
    summary = result[0]["generated_text"][len(prompt):].strip()
    return prompt, summary

# ======================
# MAIN
# ======================

# Input and output file names.
#file_path = "/Users/sanafatima/Desktop/ehr_summarization_BioMibLab-main/mimic-iv-demo/mimic-iv-clinical-database-demo-2.2/icu/icustays.csv"
#output_file = "/Users/sanafatima/Desktop/ehr_summarization_BioMibLab-main/pipelineScalingCode/icustays_prose_MG.txt"
#sql_file = "/Users/sanafatima/Desktop/ehr_summarization_BioMibLab-main/pipelineScalingCode/icustays_queries_MG.sql"
file_path = "/storage/ice1/0/2/sfatima7/icustays.csv"
output_file = "/storage/ice1/0/2/sfatima7/icustays_prose_MG.txt"
sql_file = "/storage/ice1/0/2/sfatima7/icustays_queries_MG.sql"

# Load the ICU CSV into pandas.
df = pd.read_csv(file_path)

print(f"Loaded {len(df)} rows")

# Load the CSV data into an in-memory SQLite database so the LLM-generated SQL can be executed.
conn = sqlite3.connect(":memory:")
df.to_sql("icustays", conn, index=False, if_exists="replace")

# Get the unique patient IDs we want to process.
subject_ids = df["subject_id"].dropna().astype(int).unique()
print(f"Processing {len(subject_ids)} patients")

# Open output files:
# 1. prose file for prompts + summaries
# 2. sql file for the final executed SQL queries
with open(output_file, "w") as prose_handle:
    with open(sql_file, "w") as sql_handle:
        for i, subject_id in enumerate(subject_ids):
            try:
                # Step 1: ask the LLM to generate SQL for this patient.
                sql_prompt, raw_sql_output, sql = generate_sql(subject_id)

                # Save the cleaned SQL query separately.
                sql_handle.write(f"-- Patient {subject_id}\n")
                sql_handle.write(sql)
                sql_handle.write("\n\n")

                # Step 2: execute the generated SQL against SQLite.
                result_df = pd.read_sql_query(sql, conn)

                # Step 3: convert SQL results into text context.
                context = build_patient_context(subject_id, result_df)

                # Step 4: ask the LLM to summarize the ICU stays.
                summary_prompt, summary = generate_summary(context)

                # Save everything that went into and came out of the LLM so the pipeline is easy to inspect later.
                prose_handle.write(f"=== Patient {subject_id} ===\n")
                prose_handle.write("SQL PROMPT:\n")
                prose_handle.write(sql_prompt.strip() + "\n\n")
                prose_handle.write("RAW SQL OUTPUT:\n")
                prose_handle.write(raw_sql_output + "\n\n")
                prose_handle.write("EXECUTED SQL:\n")
                prose_handle.write(sql + "\n\n")
                prose_handle.write("SUMMARY PROMPT:\n")
                prose_handle.write(summary_prompt.strip() + "\n\n")
                prose_handle.write("SUMMARY OUTPUT:\n")
                prose_handle.write(summary + "\n\n")

                print(f"Done {i+1}/{len(subject_ids)}")
            except Exception as e:
                print(f"Failed for patient {subject_id}: {e}")
                
# Close the SQLite connection once processing is done.
conn.close()

print("Finished everything.")