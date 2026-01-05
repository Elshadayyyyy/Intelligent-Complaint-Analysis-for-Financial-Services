import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

def create_chunks(df: pd.DataFrame):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )

    rows = []

    for _, row in df.iterrows():
        chunks = splitter.split_text(row["clean_text"])
        total_chunks = len(chunks)

        for i, chunk in enumerate(chunks):
            rows.append({
                "text": chunk,
                "product_category": row["product"],
                "complaint_id": row["complaint_id"],
                "chunk_index": i,
                "total_chunks": total_chunks
            })

    return pd.DataFrame(rows)


def embed_chunks(chunks_df: pd.DataFrame):
    model = SentenceTransformer("all-MiniLM-L6-v2")

    embeddings = model.encode(
        chunks_df["text"].tolist(),
        show_progress_bar=True
    )

    chunks_df["embedding"] = embeddings.tolist()
    return chunks_df


if __name__ == "__main__":
    df = pd.read_csv("../data/processed/filtered_complaints.csv")
    sample_df = (
        df.groupby("product", group_keys=False)
        .apply(lambda x: x.sample(min(len(x), 3000), random_state=42))
    )
    chunks_df = create_chunks(sample_df)
    chunks_df = embed_chunks(chunks_df)
    chunks_df.to_pickle("../data/processed/sample_chunks.pkl")
    print("Embeddings created and saved")
