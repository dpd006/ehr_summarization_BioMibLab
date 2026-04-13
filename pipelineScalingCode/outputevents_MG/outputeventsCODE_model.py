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
    oe.subject_id,
    oe.hadm_id,
    oe.stay_id,
    oe.charttime,
    oe.itemid,
    di.label,
    oe.value,
    oe.valueuom
FROM outputevents oe
LEFT JOIN d_items di
    ON oe.itemid = di.itemid
WHERE oe.subject_id = {subject_id}
ORDER BY oe.charttime ASC;
""".strip()

# ======================
# STEP 1: SQL GENERATION
# ======================

def generate_sql(subject_id):
    prompt = f"""
You are a clinical SQL assistant.

Write ONE SQLite query using:
- outputevents table columns:
  subject_id, hadm_id, stay_id, charttime, itemid, value, valueuom
- d_items table columns:
  itemid, label

Task:
Get all output events for subject_id = {subject_id}, including the output label from d_items.

Rules:
- SELECT only
- Use outputevents as the main table
- LEFT JOIN d_items ON itemid
- WHERE outputevents.subject_id = {subject_id}
- ORDER BY charttime ASC
- Return SQL only
- Do not explain anything
- Do not return Python
""".strip()
    
    result = generator(prompt, max_new_tokens=500, do_sample=False)
    raw_sql = result[0]["generated_text"][len(prompt):].strip()
    sql = extract_sql(raw_sql)

    if not sql.lower().strip().startswith("select"):
        sql = fallback_sql(subject_id)
    return prompt, raw_sql, sql

# ======================
# STEP 2: BUILD CONTEXT
# ======================

def build_patient_context(subject_id, group):
    g = group.copy()

    if "starttime" in g.columns:
        g = g.sort_values("starttime")

    summaries = []
    label_col = g["label"] if "label" in g.columns else pd.Series([None] * len(g))
    label_col = label_col.fillna("UNKNOWN_LABEL")

    for label, grp in g.groupby(label_col, dropna=False):
        hadm_id = safe_str(grp["hadm_id"].dropna().iloc[0]) if "hadm_id" in grp.columns and grp["hadm_id"].notna().any() else None
        stay_id = safe_str(grp["stay_id"].dropna().iloc[0]) if "stay_id" in grp.columns and grp["stay_id"].notna().any() else None

        first_start = safe_str(grp["starttime"].dropna().iloc[0]) if "starttime" in grp.columns and grp["starttime"].notna().any() else None
        last_end = safe_str(grp["endtime"].dropna().iloc[-1]) if "endtime" in grp.columns and grp["endtime"].notna().any() else None

        example_amount = fmt_num(grp["amount"].dropna().iloc[0]) if "amount" in grp.columns and grp["amount"].notna().any() else None
        amountuom = safe_str(grp["amountuom"].dropna().iloc[0]) if "amountuom" in grp.columns and grp["amountuom"].notna().any() else None

        example_rate = fmt_num(grp["rate"].dropna().iloc[0]) if "rate" in grp.columns and grp["rate"].notna().any() else None
        rateuom = safe_str(grp["rateuom"].dropna().iloc[0]) if "rateuom" in grp.columns and grp["rateuom"].notna().any() else None

        statuses = []
        if "statusdescription" in grp.columns:
            seen = set()
            for x in grp["statusdescription"].dropna().tolist():
                s = clean_status(x)
                if s and s not in seen:
                    seen.add(s)
                    statuses.append(s)

        summaries.append(
            {
                "label": safe_str(label),
                "count_events": int(len(grp)),
                "hadm_id": hadm_id,
                "stay_id": stay_id,
                "first_starttime": first_start,
                "last_endtime": last_end,
                "example_amount": example_amount,
                "amountuom": amountuom,
                "example_rate": example_rate,
                "rateuom": rateuom,
                "statuses_seen": statuses,
            }
        )

    # Keep prompt size stable: most frequent labels first, cap list length
    summaries.sort(key=lambda r: -(r["count_events"] or 0))
    summaries = summaries[:30]

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
You are a clinical ICU documentation assistant.

Write natural language paragraphs summarizing this patient's ICU output events in natural prose.
No bullets, no numbering, no headings, no “Field: value” labels.
Do not list events one-by-one. Synthesize patterns and trends.
Use ONLY facts in OUTPUTEVENTS_JSON. Do not invent values or times.
Do not mention SQL.
Do not hallucinate.

OUTPUTEVENTS_JSON:
{context}
""".strip()

    result = generator(prompt, max_new_tokens=500, do_sample=False)
    summary = result[0]["generated_text"][len(prompt):].strip()
    return prompt, summary


# ======================
# MAIN
# ======================

#outputevents_path = "outputevents.csv"
#d_items_path = "d_items.csv"
#output_file = "outputevents_prose.txt"
#sql_file = "outputevents_queries.sql"

outputevents_path = "/storage/ice1/0/2/sfatima7/outputevents.csv"
d_items_path = "/storage/ice1/0/2/sfatima7/d_items.csv"
output_file = "/storage/ice1/0/2/sfatima7/outputevents_prose_MG.txt"
sql_file = "/storage/ice1/0/2/sfatima7/outputevents_queries_MG.sql"

oe_df = pd.read_csv(outputevents_path)
items_df = pd.read_csv(d_items_path)

print(f"Loaded outputevents: {oe_df.shape}")
print(f"Loaded d_items: {items_df.shape}")

conn = sqlite3.connect(":memory:")
oe_df.to_sql("outputevents", conn, index=False, if_exists="replace")
items_df.to_sql("d_items", conn, index=False, if_exists="replace")

subject_ids = oe_df["subject_id"].dropna().astype(int).unique()[:5]
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
            prose_handle.write(f"=== Patient {subject_id} ===\nERROR: {e}\n\n")

conn.close()
print("Finished everything.")