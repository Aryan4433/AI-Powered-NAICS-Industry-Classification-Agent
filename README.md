# AI-Powered-NAICS-Industry-Classification-Agent


An AI-powered industry classification system that predicts NAICS 2022 codes for organizations using a Retrieval-Augmented Generation (RAG) architecture.
The system combines semantic search, vector similarity, web retrieval, and LLM reasoning to enrich company datasets with accurate industry classifications.

Overview

This project automates NAICS industry code prediction for large batches of organizations.

The pipeline works as follows:

User uploads an Excel/CSV file containing organization names.

The system retrieves relevant NAICS descriptions using FAISS vector similarity search.

It performs live web search to gather additional company context.

An LLM (via ChatGroq) analyzes the information and predicts the best NAICS 2022 code.

Results are returned as an enriched dataset that can be downloaded.

The system provides higher accuracy and reduced hallucination by combining structured NAICS data with external knowledge retrieval.

Key Features

Batch organization industry classification

Retrieval-Augmented Generation (RAG) architecture

Semantic similarity search using FAISS

LLM reasoning with ChatGroq

Real-time web information retrieval

Streamlit interface for easy interaction

Excel export of enriched results

FastAPI deployment for production APIs

System Architecture

Input Data → Web Search + Embedding Retrieval → FAISS Vector Search → LLM Reasoning → NAICS Prediction → Enriched Output

Tech Stack

Python

Streamlit – User Interface

FastAPI – API deployment

LangChain – LLM orchestration

ChatGroq (LLaMA 3.1) – Large Language Model

SentenceTransformers – Text embeddings

FAISS – Vector similarity search

DuckDuckGo Search API – Web information retrieval

Pandas / NumPy – Data processing
