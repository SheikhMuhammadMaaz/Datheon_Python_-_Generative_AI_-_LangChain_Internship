from pathlib import Path
import os
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
PDF_PATH = DATA_DIR / "Attention_Is_All_You_Need.pdf"
CHROMA_DIR = BASE_DIR / "chroma_db"


CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


TOP_K = 4


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Gemini answer-generation model
GENERATION_MODEL = "gemini-3.8-flash"

load_dotenv(BASE_DIR / ".env")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
