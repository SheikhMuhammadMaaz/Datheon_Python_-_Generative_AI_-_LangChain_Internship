from google import genai
from langchain_chroma import Chroma

from config import (
    CHROMA_DIR,
    GENERATION_MODEL,
    GEMINI_API_KEY,
    TOP_K,
)
from embeddings import LocalEmbeddings


class RAGPipeline:
    def __init__(self):
        if not GEMINI_API_KEY:
            raise ValueError(
                "GEMINI_API_KEY missing. Add it to your .env file."
            )

        if not CHROMA_DIR.exists():
            raise FileNotFoundError(
                "ChromaDB not found. Run ingest.py first."
            )

        self.client = genai.Client(api_key=GEMINI_API_KEY)

        self.vector_store = Chroma(
            collection_name="attention_paper",
            persist_directory=str(CHROMA_DIR),
            embedding_function=LocalEmbeddings(),
        )

        if self.vector_store._collection.count() == 0:
            raise ValueError(
                "ChromaDB is empty. Run ingest.py again."
            )

        self.retriever = self.vector_store.as_retriever(
            search_kwargs={"k": TOP_K}
        )

    def ask(self, question: str):

        retrieved_docs = self.retriever.invoke(question)

        if not retrieved_docs:
            return (
                "I could not find relevant information in the PDF.",
                [],
            )

        context_parts = []
        sources = []

        for doc in retrieved_docs:
            page_number = int(doc.metadata.get("page", 0)) + 1
            source_name = doc.metadata.get("source", "PDF")

            context_parts.append(
                f"[Source: {source_name}, Page: {page_number}]\n"
                f"{doc.page_content}"
            )

            source = {
                "file": source_name.split("/")[-1].split("\\")[-1],
                "page": page_number,
            }

            if source not in sources:
                sources.append(source)

        context = "\n\n".join(context_parts)

    
        prompt = f"""
You are a document question-answering assistant.

Answer the user's question using ONLY the context below.
If the context does not contain the answer, clearly say:
"I could not find this information in the provided PDF."

Do not invent facts or page numbers.
Cite supporting information using labels such as [Page 2].

CONTEXT:
{context}

USER QUESTION:
{question}

ANSWER:
"""

        response = self.client.models.generate_content(
            model=GENERATION_MODEL,
            contents=prompt,
        )

        answer = response.text

        if not answer:
            answer = "The language model returned an empty response."

        return answer, sources
