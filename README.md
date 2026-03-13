# AI-Powered-NAICS-Industry-Classification-Agent

# NAICS Industry Classification Agent

An AI-powered **industry classification assistant** that automatically predicts **NAICS (North American Industry Classification System) codes** based on company descriptions.

The system leverages **LLMs, semantic search, and embeddings** to intelligently match business activities with the most relevant NAICS categories.

This project demonstrates how **LLM agents + vector search** can be used to improve **industry data enrichment and classification accuracy**.

---
## Pipeline

The system follows a hybrid **Retrieval-Augmented Generation (RAG)** pipeline for accurate NAICS classification.

The pipeline works as follows:

1. **User Input**  
   The user uploads an **Excel or CSV file** containing organization or company names.

2. **Vector Similarity Search**  
   The system retrieves relevant **NAICS descriptions** using **FAISS vector similarity search** based on semantic embeddings.

3. **Web Context Retrieval**  
   A **live web search** is performed to gather additional contextual information about each company.

4. **LLM Reasoning**  
   An **LLM (via ChatGroq)** analyzes both the NAICS descriptions and retrieved web context to predict the **most appropriate NAICS 2022 classification code**.

5. **Data Enrichment Output**  
   The results are returned as an **enriched dataset**, including predicted NAICS codes, which users can download.

By combining **structured NAICS data**, **semantic retrieval**, and **external knowledge**, the system improves classification accuracy and reduces hallucinations compared to standalone LLM predictions.

## Features

- AI-powered **NAICS code prediction**
- **Semantic similarity search** using embeddings
- Interactive **Streamlit web interface**
- Batch processing for multiple company descriptions
- Supports **industry classification automation**
- Uses **vector search for accurate matching**
- Easily extendable for other classification systems (SIC, ISIC, NACE)

---

## Tech Stack

- Python  
- LangChain  
- Streamlit  
- Sentence Transformers  
- FAISS  
- HuggingFace Models  

Libraries used:

- `langchain`
- `sentence-transformers`
- `faiss-cpu`
- `streamlit`
- `pandas`

---

## Project Structure

