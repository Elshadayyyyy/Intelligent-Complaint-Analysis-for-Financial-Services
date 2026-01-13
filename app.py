import faiss
import pandas as pd
import numpy as np
import gradio as gr
from sentence_transformers import SentenceTransformer
from transformers import pipeline

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.read_index("vector_store/sample_faiss.index")
metadata_df = pd.read_parquet("vector_store/sample_metadata.parquet")
llm = pipeline(
    "text-generation",
    model="distilgpt2",
    max_new_tokens=120
)
def retrieve_chunks(query, k=5):
    query_embedding = embedding_model.encode([query]).astype("float32")
    distances, indices = index.search(query_embedding, k)
    results = metadata_df.iloc[indices[0]]
    texts = results["text"].tolist()
    sources = results[["product_category", "complaint_id"]].to_dict(orient="records")
    return texts, sources
def generate_answer(query, texts):
    context = "\n\n".join(texts[:3])
    prompt = f"""
You are a financial analyst assistant for CrediTrust. Use only the context below to answer the question. If the answer is not present, say you don't have enough information.
Context:
{context}
Question:
{query}
Answer:
"""
    result = llm(prompt)[0]["generated_text"]
    return result.split("Answer:")[-1].strip()
def rag_chat(query):
    try:
        texts, sources = retrieve_chunks(query)
        answer = generate_answer(query, texts)
        sources_text = "\n\n".join(
            [f"- {s['product_category']} | Complaint ID: {s['complaint_id']}" for s in sources[:3]]
        )
        return answer, sources_text
    except Exception as e:
        return f"Error: {str(e)}", "Error retrieving sources"
with gr.Blocks() as demo:
    gr.Markdown("# CrediTrust Complaint Analysis Chatbot")

    query = gr.Textbox(label="Ask a question about customer complaints")
    answer = gr.Textbox(label="AI Answer")
    sources = gr.Textbox(label="Sources")

    btn = gr.Button("Ask")

    btn.click(fn=rag_chat, inputs=query, outputs=[answer, sources])

demo.launch()
