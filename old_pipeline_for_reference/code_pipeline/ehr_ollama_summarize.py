# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# """
# EHR SQL → Natural Summary using Ollama (local LLM)
# --------------------------------------------------
# Reads a CSV with columns:
#   - NL_Question
#   - SQL_Query
#   - Answer_Structured   e.g., "(PROPOFOL,DOXEPIN HCL,...)"
# and writes a new CSV with an added "summary" column.

# Usage (PowerShell / CMD):
#     python ehr_ollama_summarize.py --csv "query1.csv" --model "llama3.2:1b"

# Dependencies:
#     pip install pandas requests

# Ollama:
#     - Install: https://ollama.com/download
#     - Pull a small model (recommended for CPU):
#         ollama pull llama3.2:1b
#     - The script calls the local API at http://127.0.ollama pull llama3.2:1b0.1:11434
# """

# import argparse
# import os
# import sys
# import time
# import json
# from typing import List, Dict, Any
# import requests
# import pandas as pd

# OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")

# # Minimal therapeutic class map (extend as needed for your lab context)
# DRUG_CLASS_MAP = {
#     # antibiotics
#     "cefepime": "antibiotic",
#     "vancomycin": "antibiotic",
#     "meropenem": "antibiotic",
#     "ceftriaxone": "antibiotic",
#     "levofloxacin": "antibiotic",
#     # cardiovascular
#     "metoprolol tartrate": "beta-blocker",
#     "amiodarone": "antiarrhythmic",
#     "heparin": "anticoagulant",
#     "simvastatin": "statin",
#     "lisinopril": "ace inhibitor",
#     "potassium chloride": "electrolyte",
#     "furosemide": "loop diuretic",
#     # psychiatric / neuro
#     "doxepin hcl": "antidepressant (TCA)",
#     "paroxetine": "antidepressant (SSRI)",
#     "lorazepam": "benzodiazepine",
#     "propofol": "sedative/anesthetic",
#     # gi
#     "famotidine": "h2 blocker",
#     "docusate sodium": "stool softener",
#     "bisacodyl": "laxative",
#     "calcium carbonate": "antacid/supplement",
#     # respiratory
#     "tiotropium bromide": "bronchodilator (anticholinergic)",
#     "albuterol-ipratropium": "bronchodilator combination",
#     # supplements
#     "cyanocobalamin": "vitamin (B12)",
#     # misc
#     "ibuprofen": "nsaid",
#     "alendronate sodium": "bisphosphonate",
#     "senna": "laxative",
#     "maalox/diphenhydramine/lidocaine": "gi cocktail",
# }

# def normalize_drug_name(name: str) -> str:
#     return name.strip().lower()

# def parse_drug_list(answer_structured: str) -> List[str]:
#     if not isinstance(answer_structured, str):
#         return []
#     text = answer_structured.strip().strip("()")
#     if not text:
#         return []
#     parts = [p.strip() for p in text.split(",") if p.strip()]
#     return parts

# def detect_classes(drugs: List[str]) -> List[str]:
#     classes = []
#     seen = set()
#     for d in drugs:
#         key = normalize_drug_name(d)
#         cls = DRUG_CLASS_MAP.get(key)
#         if cls and cls not in seen:
#             classes.append(cls)
#             seen.add(cls)
#     return classes

# def prompt_from_row(nl_question: str, drugs: List[str], classes: List[str]) -> str:
#     drug_list_str = ", ".join(drugs) if drugs else "None"
#     class_str = ", ".join(classes) if classes else "unknown"
#     return f"""You are a biomedical summarizer for an EHR system.
# Input:
# - NL question: {nl_question}
# - Prescribed medications: {drug_list_str}
# - Detected therapeutic classes: {class_str}

# Task:
# Write a concise, clinically neutral summary (1–2 sentences) of the prescription profile.
# Requirements:
# - Prefer categories (e.g., 'antibiotics', 'beta-blocker') over listing every drug.
# - Do NOT infer diagnoses or outcomes.
# - If uncertain about a drug, speak generally (e.g., 'multiple cardiovascular and antibiotic agents').
# - If no drugs are present, say 'No prescriptions found in this record.'
# - Keep it under 40 words.
# Output only the summary."""

# def call_ollama(prompt: str, model: str, timeout: float = 120.0) -> str:
#     url = f"{OLLAMA_URL}/api/generate"
#     payload: Dict[str, Any] = {
#         "model": model,
#         "prompt": prompt,
#         "stream": False,
#         "options": {
#             "temperature": 0.2,
#             "num_predict": 128,
#         },
#     }
#     try:
#         r = requests.post(url, json=payload, timeout=timeout)
#         r.raise_for_status()
#         data = r.json()
#         # Ollama returns: {"model": "...", "created_at": "...", "response": "...", ...}
#         return data.get("response", "").strip()
#     except requests.exceptions.RequestException as e:
#         raise RuntimeError(f"Ollama request failed: {e}")

# def check_ollama_running() -> bool:
#     try:
#         r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2.5)
#         return r.status_code == 200
#     except Exception:
#         return False

# def fallback_summary(drugs: List[str], classes: List[str]) -> str:
#     if not drugs:
#         return "No prescriptions found in this record."
#     if classes:
#         return f"Patient received medications spanning: {', '.join(classes)}."
#     # last resort: very compact list
#     shown = ", ".join(drugs[:5])
#     more = "" if len(drugs) <= 5 else " (and others)"
#     return f"Patient received: {shown}{more}."

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--csv", required=True, help="Path to input CSV (with NL_Question, SQL_Query, Answer_Structured).")
#     parser.add_argument("--out", default=None, help="Path to output CSV (default: <input>_with_summaries.csv).")
#     parser.add_argument("--model", default=os.environ.get("OLLAMA_MODEL", "llama3.2:1b"), help="Ollama model name.")
#     parser.add_argument("--limit", type=int, default=None, help="Only process first N rows (for quick tests).")
#     args = parser.parse_args()

#     if not os.path.exists(args.csv):
#         print(f"[ERROR] CSV not found: {args.csv}", file=sys.stderr)
#         sys.exit(1)

#     df = pd.read_csv(args.csv)
#     required_cols = {"NL_Question", "SQL_Query", "Answer_Structured"}
#     missing = required_cols - set(df.columns)
#     if missing:
#         print(f"[ERROR] Missing required columns: {missing}", file=sys.stderr)
#         sys.exit(1)

#     if args.limit is not None and args.limit > 0:
#         df = df.head(args.limit).copy()

#     # Parse and enrich
#     drug_lists = df["Answer_Structured"].apply(parse_drug_list)
#     classes = drug_lists.apply(detect_classes)

#     # Build prompts
#     prompts = [
#         prompt_from_row(nlq, drugs, cls)
#         for nlq, drugs, cls in zip(df["NL_Question"].astype(str), drug_lists, classes)
#     ]

#     # Decide output path
#     out_path = args.out or os.path.splitext(args.csv)[0] + "_with_summaries.csv"

#     # Ensure Ollama is available
#     use_ollama = check_ollama_running()
#     if not use_ollama:
#         print("[WARN] Ollama not detected at http://127.0.0.1:11434. Using fallback summaries.", file=sys.stderr)

#     # Generate summaries
#     summaries: List[str] = []
#     for i, (drugs, cls, prompt) in enumerate(zip(drug_lists, classes, prompts), start=1):
#         if use_ollama:
#             try:
#                 resp = call_ollama(prompt, model=args.model)
#             except Exception as e:
#                 print(f"[WARN] Row {i}: Ollama failed ({e}). Falling back.", file=sys.stderr)
#                 resp = fallback_summary(drugs, cls)
#         else:
#             resp = fallback_summary(drugs, cls)

#         summaries.append(resp)
#         if i % 10 == 0 or i == len(prompts):
#             print(f"[info] processed {i}/{len(prompts)} rows")

#         # brief pause to be nice to CPU
#         time.sleep(0.02)

#     df_out = df.copy()
#     df_out["drug_list"] = ["; ".join(drugs) for drugs in drug_lists]
#     df_out["drug_classes"] = [", ".join(cls) if cls else "" for cls in classes]
#     df_out["summary"] = summaries

#     df_out.to_csv(out_path, index=False, encoding="utf-8")
#     print(f"[done] wrote: {out_path}")

# if __name__ == "__main__":
#     main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EHR SQL → Natural Summary using Ollama (local LLM)
--------------------------------------------------
Reads a CSV with columns:
  - NL_Question
  - SQL_Query
  - Answer_Structured   e.g., "(PROPOFOL,DOXEPIN HCL,...)"
and writes a new CSV with an added "summary" column.

Usage:
    python ehr_ollama_summarize.py --csv "query1.csv" --model "llama3.2:1b"

Dependencies:
    pip install pandas requests
"""

import argparse
import os
import sys
import time
import json
from typing import List, Dict, Any
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")

# Minimal therapeutic class map (extend as needed)
DRUG_CLASS_MAP = {
    "cefepime": "antibiotic",
    "vancomycin": "antibiotic",
    "meropenem": "antibiotic",
    "ceftriaxone": "antibiotic",
    "levofloxacin": "antibiotic",
    "metoprolol tartrate": "beta-blocker",
    "amiodarone": "antiarrhythmic",
    "heparin": "anticoagulant",
    "simvastatin": "statin",
    "lisinopril": "ace inhibitor",
    "potassium chloride": "electrolyte",
    "furosemide": "loop diuretic",
    "doxepin hcl": "antidepressant (TCA)",
    "paroxetine": "antidepressant (SSRI)",
    "lorazepam": "benzodiazepine",
    "propofol": "sedative/anesthetic",
    "famotidine": "h2 blocker",
    "docusate sodium": "stool softener",
    "bisacodyl": "laxative",
    "calcium carbonate": "antacid/supplement",
    "tiotropium bromide": "bronchodilator (anticholinergic)",
    "albuterol-ipratropium": "bronchodilator combination",
    "cyanocobalamin": "vitamin (B12)",
    "ibuprofen": "nsaid",
    "alendronate sodium": "bisphosphonate",
    "senna": "laxative",
    "maalox/diphenhydramine/lidocaine": "gi cocktail",
}

# ---------------- Helper Functions ----------------

def normalize_drug_name(name: str) -> str:
    return name.strip().lower()

def parse_drug_list(answer_structured: str) -> List[str]:
    if not isinstance(answer_structured, str):
        return []
    text = answer_structured.strip().strip("()")
    if not text:
        return []
    return [p.strip() for p in text.split(",") if p.strip()]

def detect_classes(drugs: List[str]) -> List[str]:
    classes = []
    seen = set()
    for d in drugs:
        key = normalize_drug_name(d)
        cls = DRUG_CLASS_MAP.get(key)
        if cls and cls not in seen:
            classes.append(cls)
            seen.add(cls)
    return classes

def prompt_from_row(nl_question: str, drugs: List[str], classes: List[str]) -> str:
    drug_list_str = ", ".join(drugs) if drugs else "None"
    class_str = ", ".join(classes) if classes else "unknown"
    return f"""You are a biomedical summarizer for an EHR system.
Input:
- NL question: {nl_question}
- Prescribed medications: {drug_list_str}
- Detected therapeutic classes: {class_str}

Task:
Write a concise, clinically neutral summary (1–2 sentences) of the prescription profile.
Requirements:
- Prefer categories (e.g., 'antibiotics', 'beta-blocker') over listing every drug.
- Do NOT infer diagnoses or outcomes.
- If uncertain about a drug, speak generally (e.g., 'multiple cardiovascular and antibiotic agents').
- If no drugs are present, say 'No prescriptions found in this record.'
- Keep it under 40 words.
Output only the summary."""

def call_ollama(prompt: str, model: str, timeout: float = 120.0) -> str:
    url = f"{OLLAMA_URL}/api/generate"
    payload: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
            "num_predict": 128,
        },
    }
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        return data.get("response", "").strip()
    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"Ollama request failed: {e}")

def check_ollama_running() -> bool:
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=2.5)
        return r.status_code == 200
    except Exception:
        return False

def fallback_summary(drugs: List[str], classes: List[str]) -> str:
    if not drugs:
        return "No prescriptions found in this record."
    if classes:
        return f"Patient received medications spanning: {', '.join(classes)}."
    shown = ", ".join(drugs[:5])
    more = "" if len(drugs) <= 5 else " (and others)"
    return f"Patient received: {shown}{more}."

# ---------------- Main ----------------

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--csv", required=True)
#     parser.add_argument("--out", default=None)
#     parser.add_argument("--model", default=os.environ.get("OLLAMA_MODEL", "llama3.2:1b"))
#     parser.add_argument("--limit", type=int, default=None)
#     parser.add_argument("--workers", type=int, default=4, help="Number of parallel Ollama calls")
#     args = parser.parse_args()

#     if not os.path.exists(args.csv):
#         print(f"[ERROR] CSV not found: {args.csv}", file=sys.stderr)
#         sys.exit(1)

#     df = pd.read_csv(args.csv)
#     required_cols = {"NL_Question", "SQL_Query", "Answer_Structured"}
#     missing = required_cols - set(df.columns)
#     if missing:
#         print(f"[ERROR] Missing required columns: {missing}", file=sys.stderr)
#         sys.exit(1)

#     if args.limit is not None:
#         df = df.head(args.limit).copy()

#     drug_lists = df["Answer_Structured"].apply(parse_drug_list)
#     classes = drug_lists.apply(detect_classes)

#     prompts = [
#         prompt_from_row(nlq, drugs, cls)
#         for nlq, drugs, cls in zip(df["NL_Question"].astype(str), drug_lists, classes)
#     ]

#     out_path = args.out or os.path.splitext(args.csv)[0] + "_with_summaries.csv"
#     use_ollama = check_ollama_running()
#     if not use_ollama:
#         print("[WARN] Ollama not detected. Using fallback summaries.", file=sys.stderr)

#     summaries: List[str] = [None] * len(df)

#     if use_ollama:
#         with ThreadPoolExecutor(max_workers=args.workers) as executor:
#             future_to_idx = {executor.submit(call_ollama, prompt, args.model): i for i, prompt in enumerate(prompts)}
#             for future in as_completed(future_to_idx):
#                 i = future_to_idx[future]
#                 try:
#                     summaries[i] = future.result()
#                 except Exception as e:
#                     print(f"[WARN] Row {i+1}: Ollama failed ({e}). Falling back.", file=sys.stderr)
#                     summaries[i] = fallback_summary(drug_lists[i], classes[i])
#     else:
#         for i, (drugs, cls) in enumerate(zip(drug_lists, classes)):
#             summaries[i] = fallback_summary(drugs, cls)

#     df_out = df.copy()
#     df_out["drug_list"] = ["; ".join(drugs) for drugs in drug_lists]
#     df_out["drug_classes"] = [", ".join(cls) if cls else "" for cls in classes]
#     df_out["summary"] = summaries

#     df_out.to_csv(out_path, index=False, encoding="utf-8")
#     print(f"[done] wrote: {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--model", default=os.environ.get("OLLAMA_MODEL", "llama3.2:1b"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel Ollama calls")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"[ERROR] CSV not found: {args.csv}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.csv)
    required_cols = {"NL_Question", "SQL_Query", "Answer_Structured"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"[ERROR] Missing required columns: {missing}", file=sys.stderr)
        sys.exit(1)

    if args.limit is not None:
        df = df.head(args.limit).copy()

    drug_lists = df["Answer_Structured"].apply(parse_drug_list)
    classes = drug_lists.apply(detect_classes)

    prompts = [
        prompt_from_row(nlq, drugs, cls)
        for nlq, drugs, cls in zip(df["NL_Question"].astype(str), drug_lists, classes)
    ]

    out_path = args.out or os.path.splitext(args.csv)[0] + "_with_summaries.csv"
    use_ollama = check_ollama_running()
    if not use_ollama:
        print("[WARN] Ollama not detected. Using fallback summaries.", file=sys.stderr)

    summaries: List[str] = [None] * len(df)

    if use_ollama:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            future_to_idx = {executor.submit(call_ollama, prompt, args.model): i for i, prompt in enumerate(prompts)}
            completed_count = 0
            for future in as_completed(future_to_idx):
                i = future_to_idx[future]
                try:
                    summaries[i] = future.result()
                except Exception as e:
                    print(f"[WARN] Row {i+1}: Ollama failed ({e}). Falling back.", file=sys.stderr)
                    summaries[i] = fallback_summary(drug_lists[i], classes[i])
                completed_count += 1
                if completed_count % 10 == 0 or completed_count == len(df):
                    print(f"[info] processed {completed_count}/{len(df)} rows")
    else:
        for i, (drugs, cls) in enumerate(zip(drug_lists, classes), start=1):
            summaries[i-1] = fallback_summary(drugs, cls)
            if i % 10 == 0 or i == len(df):
                print(f"[info] processed {i}/{len(df)} rows")

    df_out = df.copy()
    df_out["drug_list"] = ["; ".join(drugs) for drugs in drug_lists]
    df_out["drug_classes"] = [", ".join(cls) if cls else "" for cls in classes]
    df_out["summary"] = summaries

    df_out.to_csv(out_path, index=False, encoding="utf-8")
    print(f"[done] wrote: {out_path}")


if __name__ == "__main__":
    main()