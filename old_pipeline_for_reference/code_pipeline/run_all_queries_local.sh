#!/bin/bash

# Local Ollama pipeline for multiple CSVs
# Make sure your venv is active and Ollama is installed locally

# List of CSVs
for i in {2..9}
do
    echo "Processing query $i..."

    input_csv="query${i}.csv"
    out1="query${i}_with_summaries.csv"
    out2="query${i}_grouped.csv"
    out3="query${i}_final.csv"

    # Step 1: generate summaries using Ollama
    python ehr_ollama_summarize.py --csv "$input_csv" --out "$out1" --model "llama3.2:1b"

    # Step 2: group patient IDs (your grouping script)
    python group_patients.py --input "$out1" --output "$out2"

    # Step 3: run second-pass summaries with your pipeline
    python summarize_groups.py --input "$out2" --output "$out3"

    echo "Finished query $i!"
done

echo "All queries done."
