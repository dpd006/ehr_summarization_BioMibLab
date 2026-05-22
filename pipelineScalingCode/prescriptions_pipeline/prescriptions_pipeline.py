import os
import pandas as pd
from transformers import pipeline


# =========================
# LOAD MODEL (NO OLLAMA)
# =========================

model = pipeline(
    "text2text-generation",
    model="google/flan-t5-base"
)


# =========================
# HELPERS
# =========================

def safe_str(value):
    if pd.isna(value) or str(value).strip() == "":
        return None
    return str(value).strip()


def build_med_context(subject_id, group):
    lines = [f"Patient ID: {subject_id}"]

    for _, row in group.iterrows():
        drug = safe_str(row.get("drug"))
        start = safe_str(row.get("starttime"))
        stop = safe_str(row.get("stoptime"))
        route = safe_str(row.get("route"))
        strength = safe_str(row.get("prod_strength"))
        dose = safe_str(row.get("dose_val_rx"))
        unit = safe_str(row.get("dose_unit_rx"))

        lines.append("\nMedication Event:")

        if drug:
            lines.append(f"- Drug: {drug}")
        if start:
            lines.append(f"- Start: {start}")
        if stop:
            lines.append(f"- Stop: {stop}")
        if route:
            lines.append(f"- Route: {route}")
        if strength:
            lines.append(f"- Strength: {strength}")
        if dose:
            lines.append(f"- Dose: {dose} {unit if unit else ''}")

    return "\n".join(lines)


# =========================
# LLM CALL
# =========================

def generate_med_summary(context):
    prompt = (
        "Summarize this medication timeline in clinical language.\n"
        "Focus on drug names, timing, dosage, and route.\n"
        "Do not hallucinate.\n\n"
        f"{context}"
    )

    result = model(
        prompt,
        max_new_tokens=200,
        do_sample=False
    )

    return result[0]["generated_text"]


# =========================
# SQL GENERATOR
# =========================

def generate_sql(subject_id):
    return (
        f"SELECT * FROM prescriptions\n"
        f"WHERE subject_id = {subject_id}\n"
        f"ORDER BY starttime ASC;"
    )


# =========================
# MAIN
# =========================

file_path = "pipelineScalingCode/data/prescriptions.csv"

df = pd.read_csv(file_path)
print(f"Loaded prescriptions: {df.shape}")
print(f"Columns: {list(df.columns)}")

grouped = df.groupby("subject_id")


output_file = "pipelineScalingCode/output/prescriptions_prose.txt"
sql_output_file = "pipelineScalingCode/output/prescriptions_queries.sql"

os.makedirs(os.path.dirname(output_file), exist_ok=True)


with open(output_file, "w", encoding="utf-8") as f:
    for i, (subject_id, group) in enumerate(grouped):

        context = build_med_context(subject_id, group)
        summary = generate_med_summary(context)

        f.write(f"=== Patient {subject_id} ===\n")
        f.write(summary + "\n\n")

        print(f"[{i+1}/{len(grouped)}] Done {subject_id}")


with open(sql_output_file, "w", encoding="utf-8") as f:
    for subject_id in df["subject_id"].unique():
        f.write(generate_sql(subject_id) + "\n\n")

print("DONE")