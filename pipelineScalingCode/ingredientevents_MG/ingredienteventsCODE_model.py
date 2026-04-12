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


def fmt_num(value, digits=2):
    if value is None or pd.isna(value):
        return None
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value).strip()


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
    ie.subject_id,
    ie.hadm_id,
    ie.stay_id,
    ie.itemid,
    di.label,
    ie.starttime,
    ie.endtime,
    ie.amount,
    ie.amountuom,
    ie.rate,
    ie.rateuom,
    ie.originalrate,
    ie.statusdescription
FROM ingredientevents ie
LEFT JOIN d_items di
    ON ie.itemid = di.itemid
WHERE ie.subject_id = {subject_id}
ORDER BY ie.starttime ASC;
""".strip()


# ======================
# STEP 1: SQL GENERATION
# ======================

def generate_sql(subject_id):
    prompt = f"""
You are a clinical SQL assistant.

Write ONE SQLite query using:
- ingredientevents table with columns:
  subject_id, hadm_id, stay_id, itemid, starttime, endtime, amount, amountuom,
  rate, rateuom, originalrate, statusdescription
- d_items table with columns:
  itemid, label

Task:
Get all ingredient events for subject_id = {subject_id}, including the medication label from d_items.

Rules:
- SELECT only
- Use ingredientevents as the main table
- LEFT JOIN d_items ON itemid
- WHERE ingredientevents.subject_id = {subject_id}
- ORDER BY starttime ASC
- Return SQL only
- Do not explain anything
- Do not return Python
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
        lines.append("\nMedication Event:")

        hadm_id = safe_str(row.get("hadm_id"))
        stay_id = safe_str(row.get("stay_id"))
        label = safe_str(row.get("label"))
        itemid = safe_str(row.get("itemid"))
        starttime = safe_str(row.get("starttime"))
        endtime = safe_str(row.get("endtime"))
        amount = fmt_num(row.get("amount"))
        amountuom = safe_str(row.get("amountuom"))
        rate = fmt_num(row.get("rate"))
        rateuom = safe_str(row.get("rateuom"))
        originalrate = fmt_num(row.get("originalrate"))
        status = clean_status(row.get("statusdescription"))

        if hadm_id:
            lines.append(f"Hospital Admission ID: {hadm_id}")
        if stay_id:
            lines.append(f"ICU Stay ID: {stay_id}")
        if label:
            lines.append(f"Medication label: {label}")
        if itemid:
            lines.append(f"Medication item ID: {itemid}")
        if starttime:
            lines.append(f"Start time: {starttime}")
        if endtime:
            lines.append(f"End time: {endtime}")
        if amount and amountuom:
            lines.append(f"Recorded amount: {amount} {amountuom}")
        if rate and rateuom:
            lines.append(f"Infusion rate: {rate} {rateuom}")
        if status and "rate" in status and originalrate:
            lines.append(f"Original infusion rate: {originalrate} {rateuom or ''}".strip())
            lines.append(f"Status: {status}")
        elif status:
            lines.append(f"Status: {status}")

    return "\n".join(lines)
"""

def build_patient_context(subject_id, group):
    g = group.copy()
    if "starttime" in g.columns:
        g = g.sort_values("starttime")

    # Aggregate by medication label so we summarize ALL events without blowing context length.
    summaries = []
    for label, grp in g.groupby(g.get("label").fillna("UNKNOWN_LABEL")):
        rec = {
            "label": safe_str(label),
            "count_events": int(len(grp)),
            "hadm_id": safe_str(grp["hadm_id"].dropna().iloc[0]) if "hadm_id" in grp and grp["hadm_id"].notna().any() else None,
            "stay_id": safe_str(grp["stay_id"].dropna().iloc[0]) if "stay_id" in grp and grp["stay_id"].notna().any() else None,
            "first_starttime": safe_str(grp["starttime"].dropna().iloc[0]) if "starttime" in grp and grp["starttime"].notna().any() else None,
            "last_endtime": safe_str(grp["endtime"].dropna().iloc[-1]) if "endtime" in grp and grp["endtime"].notna().any() else None,
            "example_amount": fmt_num(grp["amount"].dropna().iloc[0]) if "amount" in grp and grp["amount"].notna().any() else None,
            "amountuom": safe_str(grp["amountuom"].dropna().iloc[0]) if "amountuom" in grp and grp["amountuom"].notna().any() else None,
            "example_rate": fmt_num(grp["rate"].dropna().iloc[0]) if "rate" in grp and grp["rate"].notna().any() else None,
            "rateuom": safe_str(grp["rateuom"].dropna().iloc[0]) if "rateuom" in grp and grp["rateuom"].notna().any() else None,
            "statuses_seen": sorted({clean_status(x) for x in grp.get("statusdescription", pd.Series([])).dropna().tolist() if clean_status(x)}),
        }
        summaries.append(rec)

    payload = {
        "patient_id": int(subject_id),
        "ingredient_event_summary_by_label": summaries,
    }
    return json.dumps(payload, ensure_ascii=True)

# ======================
# STEP 3: SUMMARY GENERATION
# ======================

def generate_summary(context):
    prompt = f"""
You are a clinical medication documentation assistant.

Write natural language paragraph summarizing this patient's ICU ingredient events.
No bullets, no numbering, no headings, no “Field: value” labels.
Do not list every event. Synthesize patterns across the data.
Use ONLY facts in INGREDIENT_EVENTS_JSON. Do not invent times, labels, or IDs.
Do not mention SQL.
Do not hallucinate.

INGREDIENT_EVENTS_JSON:
{context}
""".strip()

    result = generator(prompt, max_new_tokens=500, do_sample=False)
    summary = result[0]["generated_text"][len(prompt):].strip()
    return prompt, summary



# ======================
# MAIN
# ======================

#ingredientevents_path = "ingredientevents.csv"
#d_items_path = "d_items.csv"
#output_file = "ingredientevents_prose.txt"
#sql_file = "ingredientevents_queries.sql"

ingredientevents_path = "/storage/ice1/0/2/sfatima7/ingredientevents.csv"
d_items_path = "/storage/ice1/0/2/sfatima7/d_items.csv"
output_file = "/storage/ice1/0/2/sfatima7/ingredientevents_prose_MG.txt"
sql_file = "/storage/ice1/0/2/sfatima7/ingredientevents_queries_MG.sql"

ingredient_df = pd.read_csv(ingredientevents_path)
d_items_df = pd.read_csv(d_items_path)

print(f"Loaded ingredientevents: {ingredient_df.shape}")
print(f"Loaded d_items: {d_items_df.shape}")

conn = sqlite3.connect(":memory:")
ingredient_df.to_sql("ingredientevents", conn, index=False, if_exists="replace")
d_items_df.to_sql("d_items", conn, index=False, if_exists="replace")

subject_ids = ingredient_df["subject_id"].dropna().astype(int).unique()
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
