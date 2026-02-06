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
        "options": {
            "temperature": 0.2,  # Lower temperature for more structured/factual output
            "num_predict": 400   # Increased token limit for longer timelines
        }
    }, timeout=120)
    return r.json().get("response", "").strip()

# --- Extract numeric admission id ---
def extract_admission_id(text: str):
    """
    Extracts admission ID (HADM_ID or 'ADMISSION ID') from a string.
    """
    if not isinstance(text, str):
        return None
    match = re.search(r"(HADM_ID|ADMISSION ID)\s*=?\s*(\d+)", text, re.IGNORECASE)
    if match:
        return int(match.group(2))
    return None

def main(input_csv, output_csv):
    df = pd.read_csv(input_csv)

    # Derive PATIENT_ID
    df["PATIENT_ID"] = df.apply(
        lambda row: extract_admission_id(str(row.get("SQL_Query"))) or
                    extract_admission_id(str(row.get("NL_Question"))),
        axis=1
    )

    if df["PATIENT_ID"].isna().all():
        raise ValueError("Could not extract any PATIENT_ID values. Check your input formatting.")

    # Group all row-level summaries per patient
    # We join them with newlines to help the model distinguish events
    grouped = df.groupby("PATIENT_ID")["summary"].apply(
        lambda s: "\n- ".join(str(x) for x in s if pd.notna(x))
    )

    results = []
    use_ollama = check_ollama_running()

    for pid, text in grouped.items():
        if use_ollama:
            # --- UPDATED PROMPT FOR LIFELINES TEMPLATE ---
            prompt = f"""You are a clinical summarizer specializing in longitudinal patient history.

Input Data (Individual Events):
- {text}

Task: Reconstruct this patient's history into a "Lifelines" visualization format.
1. ORDER events chronologically (Past -> Present) if time clues exist.
2. CATEGORIZE each event into one of: [Diagnoses], [Medications], [Labs], [Procedures].
3. MERGE related items (e.g., combine multiple antibiotic prescriptions into one entry).

Required Output Format:
[Category] | [Event Description]

Example:
[Diagnoses] | Admitted for Pneumonia
[Medications] | Prescribed Vancomycin and Cefepime
[Labs] | Low potassium levels detected

Start directly with the summary:
"""
            try:
                summary = call_ollama(prompt, MODEL)
            except Exception:
                summary = text  # fallback
        else:
            summary = text

        results.append({
            "PATIENT_ID": pid,
            "lifelines_summary": summary
        })

    out_df = pd.DataFrame(results)
    out_df.to_csv(output_csv, index=False)
    print(f"[done] wrote: {output_csv}")

if __name__ == "__main__":
    # Make sure to point to your actual input file from Step 1
    main("query1_with_summaries.csv", "patient_lifelines_summaries.csv")