import os
from dotenv import load_dotenv

load_dotenv()

NLP_MODE        = os.getenv("NLP_MODE", "local")
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "")
OPENAI_EMBED    = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-ada-002")
OPENAI_CHAT     = os.getenv("OPENAI_CHAT_MODEL", "gpt-3.5-turbo")

TOP_K           = int(os.getenv("TOP_K_RESULTS", 5))
CHUNK_SIZE      = int(os.getenv("CHUNK_SIZE", 512))
CHUNK_OVERLAP   = int(os.getenv("CHUNK_OVERLAP", 64))

FLASK_PORT      = int(os.getenv("FLASK_PORT", 5000))
FLASK_DEBUG     = os.getenv("FLASK_DEBUG", "true").lower() == "true"

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR      = os.path.join(BASE_DIR, "uploads")
INDEX_DIR       = os.path.join(BASE_DIR, "index")
DB_PATH         = os.path.join(INDEX_DIR, "docsy.db")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(INDEX_DIR, exist_ok=True)
