import pandas as pd
import requests
import os
import re

# configurable
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:1b")

def check_ollama_running() -> bool:
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
        return r.status_code == 200
    except Exception:
        return False

def call_ollama(prompt: str, model: str) -> str:
    r = requests.post(f"{OLLAMA_URL}/api/generate", json={
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.3, "num_predict": 200}
    }, timeout=120)
    return r.json().get("response", "").strip()

# --- NEW: extract numeric admission id from SQL or NL_Question ---
def extract_admission_id(text: str):
    """
    Extracts admission ID (HADM_ID or 'ADMISSION ID') from a string like:
    'SELECT ... WHERE PRESCRIPTIONS.HADM_ID = 113333'
    """
    if not isinstance(text, str):
        return None
    match = re.search(r"(HADM_ID|ADMISSION ID)\s*=?\s*(\d+)", text, re.IGNORECASE)
    if match:
        return int(match.group(2))
    return None

def main(input_csv, output_csv):
    df = pd.read_csv(input_csv)

    # Derive PATIENT_ID from either SQL_Query or NL_Question
    df["PATIENT_ID"] = df.apply(
        lambda row: extract_admission_id(str(row.get("SQL_Query"))) or
                    extract_admission_id(str(row.get("NL_Question"))),
        axis=1
    )

    if df["PATIENT_ID"].isna().all():
        raise ValueError("Could not extract any PATIENT_ID values. Check your input formatting.")

    # group all summaries per patient
    grouped = df.groupby("PATIENT_ID")["summary"].apply(
        lambda s: " ".join(str(x) for x in s if pd.notna(x))
    )

    results = []
    use_ollama = check_ollama_running()

    for pid, text in grouped.items():
        if use_ollama:
            prompt = f"""You are a clinical summarization model.
Given the following individual record summaries for one patient, write a concise, neutral summary (max 3 sentences).

Records:
{text}
"""
            try:
                summary = call_ollama(prompt, MODEL)
            except Exception:
                summary = text  # fallback
        else:
            summary = text

        results.append({
            "PATIENT_ID": pid,
            "combined_summary": summary
        })

    out_df = pd.DataFrame(results)
    out_df.to_csv(output_csv, index=False)
    print(f"[done] wrote: {output_csv}")

if __name__ == "__main__":
    main("query1_with_summaries.csv", "patient_level_summaries.csv")