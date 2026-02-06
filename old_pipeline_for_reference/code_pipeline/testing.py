from transformers import pipeline
import pandas as pd

# Load your data into a DataFrame
df = pd.read_csv('query1.csv')  # Replace with your actual data file path

summarizer = pipeline(
    "summarization",
    model="sshleifer/distilbart-cnn-12-6",  # small, fast
    device_map="auto"
)

texts = df["Answer_Structured"].apply(
    lambda x: f"The patient was prescribed: {x.strip('()')}."
).tolist()

summaries = summarizer(
    texts,
    batch_size=4,
    truncation=True,
    max_length=25,
    min_length=10,
    do_sample=False
)


df["summary"] = df["Answer_Structured"].apply(
    lambda x: f"This patient received multiple medications including {x.split(',')[0].strip('()')} and others."
)
