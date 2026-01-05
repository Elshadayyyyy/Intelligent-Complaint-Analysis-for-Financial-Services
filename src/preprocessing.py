import pandas as pd
import re

ALLOWED_PRODUCTS = [
    "Credit card",
    "Personal loan",
    "Savings account",
    "Money transfer"
]

def clean_text(text: str) -> str:
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
def preprocess_complaints(
    input_path: str,
    output_path: str
):
    df = pd.read_csv(input_path)
    df = df[df["product"].isin(ALLOWED_PRODUCTS)]
    df = df.dropna(subset=["consumer_complaint_narrative"])
    df["clean_text"] = df["consumer_complaint_narrative"].apply(clean_text)

    df = df[df["clean_text"].str.len() > 20]

    df.to_csv(output_path, index=False)
    print(f"Saved cleaned data to {output_path}")

if __name__ == "__main__":
    preprocess_complaints(
        input_path="../data/raw/complaints.csv",
        output_path="../data/processed/filtered_complaints.csv"
    )
