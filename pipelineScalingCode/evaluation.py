import re
import numpy as np
import pandas as pd
from rouge_score import rouge_scorer
from bert_score import score

def normalize_text(text):

    text = text.lower()                      # remove case differences
    text = re.sub(r'select .*?;', '', text, flags=re.DOTALL)
    text = re.sub(r'\s+', ' ', text)         # remove extra spaces
    text = re.sub(r'[^\w\s\-:]', '', text)   # remove punctuation
    text = text.strip()

    return text

print("Starting per-patient evaluation...")

# -----------------------------
# Load files
# -----------------------------
with open("/Users/sanafatima/Desktop/ehr_summarization_BioMibLab-main/pipelineScalingCode/admissions_proseLLM.txt") as f:
    llm_text = f.read()

with open("/Users/sanafatima/Desktop/ehr_summarization_BioMibLab-main/pipelineScalingCode/admissions_proseM.txt") as f:
    manual_text = f.read()

print("Files loaded")

# -----------------------------
# Split summaries by patient
# -----------------------------
def split_patients(text):

    patients = {}
    blocks = re.split(r'Patient (\d+)', text)

    for i in range(1, len(blocks), 2):
        pid = blocks[i]
        summary = blocks[i+1].strip()
        patients[pid] = summary

    return patients


llm_patients = split_patients(llm_text)
manual_patients = split_patients(manual_text)

print("LLM patients:", len(llm_patients))
print("Manual patients:", len(manual_patients))

# -----------------------------
# Match patient IDs
# -----------------------------
common_patients = set(llm_patients.keys()).intersection(manual_patients.keys())
common_patients = sorted(common_patients)[:5]

print("Matched patients:", len(common_patients))

# -----------------------------
# Initialize metrics
# -----------------------------
scorer = rouge_scorer.RougeScorer(
    ['rouge1','rouge2','rougeL'],
    use_stemmer=True
)

results = []

# -----------------------------
# Evaluate each patient
# -----------------------------
for pid in common_patients:

    print("Evaluating patient:", pid)
    ref = normalize_text(manual_patients[pid])
    pred = normalize_text(llm_patients[pid])

    rouge = scorer.score(ref, pred)

    rouge1 = rouge['rouge1'].fmeasure
    rouge2 = rouge['rouge2'].fmeasure
    rougeL = rouge['rougeL'].fmeasure

    P, R, F1 = score([pred], [ref], lang="en")
    bert_f1 = F1.mean().item()

    results.append({
        "patient_id": pid,
        "rouge1": rouge1,
        "rouge2": rouge2,
        "rougeL": rougeL,
        "bertscore_f1": bert_f1
    })

    print(f"Evaluated patient {pid}")

# -----------------------------
# Convert to dataframe
# -----------------------------
df = pd.DataFrame(results)

# -----------------------------
# Save CSV
# -----------------------------
df.to_csv("evaluation_results.csv", index=False)

print("\nSaved results to evaluation_results.csv")

# -----------------------------
# Compute averages
# -----------------------------
print("\n==== AVERAGE RESULTS ====")

print("Average ROUGE-1:", df["rouge1"].mean())
print("Average ROUGE-2:", df["rouge2"].mean())
print("Average ROUGE-L:", df["rougeL"].mean())
print("Average BERTScore F1:", df["bertscore_f1"].mean())





'''
from rouge_score import rouge_scorer
from bert_score import score
import pandas as pd

# Load files
with open("/Users/sanafatima/Desktop/ehr_summarization_BioMibLab-main/pipelineScalingCode/admissions_proseLLM.txt") as f:
    llm_prose = f.read()

with open("/Users/sanafatima/Desktop/ehr_summarization_BioMibLab-main/pipelineScalingCode/admissions_proseM.txt") as f:
    manual_prose = f.read()

with open("/Users/sanafatima/Desktop/ehr_summarization_BioMibLab-main/pipelineScalingCode/admissions_queriesLLM.sql") as f:
    llm_sql = f.read()

with open("/Users/sanafatima/Desktop/ehr_summarization_BioMibLab-main/pipelineScalingCode/admissions_queriesM.sql") as f:
    manual_sql = f.read()

### ROUGE Evalution
scorer = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'], use_stemmer=True)

# summaries
rouge_prose = scorer.score(manual_prose, llm_prose)

# sql
rouge_sql = scorer.score(manual_sql, llm_sql)

print("ROUGE (Summaries):", rouge_prose)
print("ROUGE (SQL):", rouge_sql)

### BERTScore Evaluation
# summaries
P, R, F1 = score([llm_prose], [manual_prose], lang="en", rescale_with_baseline=True)

print("BERTScore Precision:", P.mean().item())
print("BERTScore Recall:", R.mean().item())
print("BERTScore F1:", F1.mean().item())

# sql 
P_sql, R_sql, F1_sql = score([llm_sql], [manual_sql], lang="en")

print("SQL BERTScore F1:", F1_sql.mean().item())

'''