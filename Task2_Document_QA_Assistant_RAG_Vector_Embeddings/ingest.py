from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from config import (
    PDF_PATH,
    CHROMA_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)
from embeddings import LocalEmbeddings


def ingest_documents():
    
    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"PDF not found: {PDF_PATH}\n"
            "Put the PDF inside the data folder."
        )

    print("Loading PDF...")
    loader = PyPDFLoader(str(PDF_PATH))
    documents = loader.load()

    if not documents:
        raise ValueError("No pages or text could be loaded from the PDF.")

    print(f"Pages loaded: {len(documents)}")

    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)

    if not chunks:
        raise ValueError("No text chunks were created from the PDF.")

    print(f"Chunks created: {len(chunks)}")

    
    print("Loading local embedding model...")
    embeddings = LocalEmbeddings()

    print("Creating embeddings and saving to ChromaDB...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_name="attention_paper",
    )

    print("Ingestion completed successfully!")
    print(f"Vector database location: {CHROMA_DIR}")
    print(f"Stored chunks: {vector_store._collection.count()}")


if __name__ == "__main__":
    ingest_documents()
