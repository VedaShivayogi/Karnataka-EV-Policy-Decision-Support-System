"""
config.py
Central configuration for the Karnataka EV Policy Decision Support System.

Set your free-tier API keys as environment variables (recommended) or
fill them in directly below for local testing.

Free key sources:
- GROQ_API_KEY   -> https://console.groq.com  (Llama 3.1/3.3, Mistral, Gemma - free tier)
- GOOGLE_API_KEY -> https://aistudio.google.com (Gemini 2.5 Flash - free tier)
- NEWS_API_KEY   -> https://newsapi.org (free developer tier, 100 req/day)
"""

import os

# Load variables from a local .env file if python-dotenv is installed and a
# .env file exists. This keeps real API keys out of source code and out of
# chat/version control. Copy .env.example to .env and fill in your own keys.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ---------- LLM Provider ----------
# Choose "groq", "gemini", or "ollama" (local, no key needed)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# ---------- News ----------
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
NEWS_QUERY = "electric vehicle policy India OR battery technology OR EV subsidy"
NEWS_PAGE_SIZE = 15

# ---------- Embeddings ----------
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# ---------- Vector store ----------
VECTOR_DB_PATH = os.getenv("VECTOR_DB_PATH", "data/vector_store")

# ---------- Voice ----------
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")   # tiny/base/small/medium
TTS_ENGINE = os.getenv("TTS_ENGINE", "piper")                  # "piper" or "coqui"
PIPER_VOICE_MODEL = os.getenv("PIPER_VOICE_MODEL", "en_US-lessac-medium")

# ---------- Paths ----------
DATA_DIR = "data"
OUTPUT_DIR = "outputs"
SCENARIO_DIR = "scenarios"
