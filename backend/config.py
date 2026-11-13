import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ---------- Paths ----------
BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
DB_DIR = BASE_DIR / "db"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
DB_DIR.mkdir(parents=True, exist_ok=True)

# ---------- OpenAI ----------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"

# ---------- RAG settings ----------
CHUNK_SIZE = 600        # characters per chunk (resumes are short)
CHUNK_OVERLAP = 100     # overlap so sentences aren't cut off
TOP_K = 4               # chunks retrieved per question
MAX_REWRITES = 2        # max query rewrites in the graph

# ---------- Semantic cache ----------
CACHE_ENABLED = True
CACHE_DISTANCE_THRESHOLD = 0.10   # cosine distance; 0 = identical, lower = stricter

# ---------- Upload limits ----------
MAX_FILE_MB = 5

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found. Add it to your .env file.")