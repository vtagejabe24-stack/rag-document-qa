"""Central configuration for the RAG Document QA system."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DOCS_DIR = ROOT / "data" / "documents"          # uploaded / sample documents
EVAL_FILE = ROOT / "data" / "evaluation" / "qa.jsonl"
INDEX_DIR = ROOT / "vector_db"                  # persisted embeddings + metadata
RESULTS_DIR = ROOT / "results"                  # evaluation / experiment output

# QASPER benchmark (created by prepare_qasper.py)
QASPER_DIR = ROOT / "data" / "qasper"
QASPER_DOCS_DIR = QASPER_DIR / "documents"
QASPER_EVAL_FILE = QASPER_DIR / "qa.jsonl"
QASPER_INDEX_DIR = ROOT / "vector_db_qasper"
QASPER_FULL_DIR = ROOT / "data" / "qasper_full"   # whole validation split (prepare_qasper.py --full)

DATASETS = {   # name -> (documents dir, eval file, index dir)
    "sample": (DOCS_DIR, EVAL_FILE, INDEX_DIR),
    "qasper": (QASPER_DOCS_DIR, QASPER_EVAL_FILE, QASPER_INDEX_DIR),
    "qasper_full": (QASPER_FULL_DIR / "documents", QASPER_FULL_DIR / "qa.jsonl", ROOT / "vector_db_qasper_full"),
}

MAX_UPLOAD_MB = 50

# --- Deep learning models (Hugging Face) ---
EMBED_MODEL = "BAAI/bge-small-en-v1.5"                  # bi-encoder, 384-dim, 512-token limit
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"   # cross-encoder reranker
# Offline fallback generator. FLAN-T5 answers in ~10 s on a CPU. An instruction-tuned
# chat model such as Qwen/Qwen2.5-0.5B-Instruct scored higher Answer-F1 on QASPER
# (0.28 vs 0.22, 30 questions) but took ~100 s per answer, so it is opt-in.
LOCAL_LLM = os.environ.get("RAG_LOCAL_LLM", "google/flan-t5-base")

# --- Chunking (measured in tokens of the embedding model's tokenizer) ---
CHUNK_SIZE = 256
CHUNK_OVERLAP = 40

# --- Retrieval ---
TOP_K_RETRIEVE = 20     # candidates from FAISS (input to the reranker)
TOP_K_CONTEXT = 5       # chunks passed to the LLM

# --- Optional features: OFF by default so the core system stays simple ---
# Core: upload -> extract -> chunk -> embed -> FAISS -> retrieve -> LLM -> answer + sources.
# Flip these once the core works (or set e.g. RAG_RERANK=1 in the environment).
def _flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default) == "1"

ENABLE_RERANKER = _flag("RAG_RERANK")           # cross-encoder reranking of retrieved chunks
# BM25 keyword search fused with FAISS. On by default: on full QASPER it raised every
# metric (Recall@10 0.839 -> 0.866) for ~5 ms extra per query.
ENABLE_HYBRID = _flag("RAG_HYBRID", "1")
# How many retrieved chunks the cross-encoder rescores. 10 matched 20 on MRR on full
# QASPER (0.474 vs 0.470) at lower latency.
RERANK_CANDIDATES = 10
ENABLE_QUERY_REWRITE = _flag("RAG_REWRITE")     # turn follow-ups into standalone queries (needs Claude)
ENABLE_MEMORY = _flag("RAG_MEMORY")             # send chat history to the LLM (needs Claude)
ENABLE_STUDY_TOOLS = _flag("RAG_STUDY_TOOLS")   # summaries, MCQs, document comparison (needs Claude)
ENABLE_ACCOUNTS = _flag("RAG_ACCOUNTS")         # per-user sign-up/login with private libraries
USERS_FILE = ROOT / "users" / "users.json"      # scrypt-hashed passwords
USERS_DIR = ROOT / "users" / "data"             # one private documents + index folder per user

# --- Generation ---
# "auto" = Claude when credentials are configured, otherwise the local FLAN-T5 model.
LLM_BACKEND = os.environ.get("RAG_LLM", "auto")          # auto | claude | local
CLAUDE_MODEL = "claude-opus-5-5"
CLAUDE_EFFORT = "medium"

# --- Experiments (Phase 16) ---
EXPERIMENT_EMBED_MODELS = [
    "sentence-transformers/all-MiniLM-L6-v2",    # 256-token limit
    "BAAI/bge-small-en-v1.5",                    # 512-token limit
    "sentence-transformers/all-mpnet-base-v2",   # 384-token limit
]
EXPERIMENT_CHUNK_SIZES = [128, 256, 384, 500]
