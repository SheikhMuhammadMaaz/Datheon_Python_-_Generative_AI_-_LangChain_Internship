from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings

from config import EMBEDDING_MODEL


class LocalEmbeddings(Embeddings):
    """Create text embeddings locally using Sentence Transformers."""

    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        return [vector.tolist() for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        vector = self.model.encode(
            text,
            normalize_embeddings=True,
        )
        return vector.tolist()
