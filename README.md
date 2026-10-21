# Datheon Python + Generative AI & LangChain Internship

This repository contains two projects completed as part of the Datheon internship.

## Task 1: Structured LLM Prompt Pipeline and JSON Extractor

- Uses structured prompts for LLM-based information extraction.
- Generates structured JSON output.
- Uses Pydantic for output validation.

## Task 2: Document Q&A Assistant with RAG and Vector Embeddings

- Loads and processes PDF documents.
- Splits documents into smaller chunks.
- Generates embeddings using Sentence Transformers.
- Stores and retrieves relevant chunks using ChromaDB.
- Uses Gemini to answer questions based on retrieved context.
- Includes source-page references in answers.

## Project Structure

- `Task1_Structured_LLM_Prompt_Pipeline_JSON_Extractor/`
- `Task2_Document_QA_Assistant_RAG_Vector_Embeddings/`

## Setup

Install dependencies from the `requirements.txt` file inside the relevant task folder.

Configure your Gemini API key locally using a `.env` file if required. Never upload API keys, `.env` files, or virtual environments to GitHub.

## Purpose

These projects demonstrate structured LLM output extraction and retrieval-augmented question answering.
