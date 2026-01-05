import streamlit as st
import faiss
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
st.set_page_config(page_title="CrediTrust Complaint Assistant")
@st.cache_resource
def load_resources():
    index = faiss.read_index("vector_store/sample_faiss.index")
    metadata = pd.read_csv("vector_store/sample_metadata.csv")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return index, metadata, model
index, metadata, model = load_resources()
st.title("CrediTrust Complaint Analysis Assistant")
query = st.text_input("Ask a question about customer complaints:")
if st.button("Ask") and query:
    q_embedding = model.encode([query]).astype("float32")
    D, I = index.search(q_embedding, k=5)
    st.subheader("Retrieved Complaint Excerpts")
    for idx in I[0]:
        st.markdown(f"- {metadata.iloc[idx]['text'][:300]}...")
