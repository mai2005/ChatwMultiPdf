import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

EMBEDDING_MODEL = "models/gemini-embedding-001"
LLM_MODEL = "gemini-2.5-flash"
RERANK_MODEL = "gemini-2.0-flash"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

FAISS_PATH = "faiss_index"
