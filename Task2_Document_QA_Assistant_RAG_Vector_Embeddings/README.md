# Attention Is All You Need - RAG Pipeline

## Overview

A Retrieval-Augmented Generation (RAG) application that
answers questions using the research paper "Attention Is
All You Need".

## Features

- PDF text extraction with page metadata
- Recursive text chunking with overlap
- Local embeddings using Sentence Transformers
- Persistent vector storage with ChromaDB
- Semantic retrieval of relevant passages
- Gemini-powered answers grounded in retrieved context
- Retrieved source pages displayed with answers

## Tech Stack

- Python
- LangChain
- PyPDF
- Sentence Transformers
- ChromaDB
- Google Gemini API

## Setup

1. Create and activate a Python virtual environment.
2. Install dependencies:

   `pip install -r requirements.txt`

3. Place the PDF at:

   `data/Attention_Is_All_You_Need.pdf`

4. Create a `.env` file and set `GEMINI_API_KEY`.

5. Build the vector database:

   `python ingest.py`

6. Start the question-answering interface:

   `python main.py`

## Architecture

PDF -> Text Extraction -> Chunking -> Local Embeddings
-> ChromaDB -> Retrieval -> Gemini -> Answer + Sources

## Notes

The embedding model runs locally. Answer generation
requires access to the configured Gemini API model.
