# Intelligent Complaint Analysis for Financial Services  
## Project Overview

CrediTrust Financial is a fast-growing digital financial services company operating in East African markets. With over 500,000 users across multiple countries, the company receives thousands of customer complaints each month through in-app submissions, email, and regulatory portals.

These complaints are rich in insight but unstructured, scattered, and time-consuming to analyze manually.
This project builds an Intelligent Complaint Analysis System using Retrieval-Augmented Generation (RAG) to transform raw complaint narratives into searchable, evidence-backed insights for internal teams such as Product, Support, and Compliance.

## Business Objectives

The system is designed to:

- Reduce the time required to identify major complaint trends from days to minutes
- Enable non technical teams to ask natural-language questions and get clear answers
- Shift the organization from reactive to proactive issue identification
- Provide transparent, source backed answers grounded in real customer complaints

---

## Solution Summary

The solution follows:

1. Complaint narratives are cleaned and processed
2. Text is chunked into manageable segments
3. Each chunk is converted into a semantic vector embedding
4. Vectors are stored in a FAISS vector database
5. User questions are embedded and matched against the database
6. Retrieved complaint chunks are passed to an LLM for grounded answer generation

---

##  Project Structure

```text
│
├── data/                   
│   └── processed/              
│
├── vector_store/              
│
├── notebooks/
│   ├── task1_eda_preprocessing.ipynb
│   └── task2_chunking_embeddings.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── embeddings.py
│   └── vector_store.py
│
├── app.py                      
├── requirements.txt
├── README.md
└── .gitignore
