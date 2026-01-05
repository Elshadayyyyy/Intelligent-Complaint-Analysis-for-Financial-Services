import faiss
import numpy as np
import pandas as pd
import os

def build_faiss_index(chunks_df: pd.DataFrame, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    embeddings = np.vstack(chunks_df["embedding"].values).astype("float32")
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, f"{save_dir}/sample_faiss.index")
    metadata = chunks_df.drop(columns=["embedding"])
    metadata.to_csv(f"{save_dir}/sample_metadata.csv", index=False)
    print(f"FAISS index saved with {index.ntotal} vectors")
if __name__ == "__main__":
    chunks_df = pd.read_pickle("../data/processed/sample_chunks.pkl")
    build_faiss_index(chunks_df, "../vector_store")
