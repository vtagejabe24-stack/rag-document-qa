# RAG-Based Document Question Answering System

Upload documents (PDF, DOCX, PPTX, TXT, MD), ask questions, and get an answer **with the source text and page/section it came from**. Built from the components up (no LangChain) and evaluated on the **QASPER** dataset (questions about NLP research papers).

```
Document -> extract text -> chunk -> Sentence-BERT embeddings -> FAISS
Question -> embed -> FAISS top-k chunks -> LLM -> answer + sources
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:ANTHROPIC_API_KEY = "sk-ant-..."     # optional - without it a local FLAN-T5 model answers
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The app opens at http://localhost:8501 (use `--server.port 8502` if that port is busy). Sample documents in `data/documents/` are indexed on first launch. Embeddings are saved in `vector_db/`, so unchanged files are never re-embedded. The embedding and FLAN-T5 models are downloaded from Hugging Face on first run and cached, so later runs work offline.

## Core modules (`rag/`)

| File | Role |
|---|---|
| `loader.py` | Text extraction with metadata (file, page/slide, section) |
| `chunker.py` | Sentence-aware chunks of ~256 tokens with overlap; each chunk knows its pages/sections |
| `embeddings.py` | Pretrained Sentence-BERT-style encoder (`BAAI/bge-small-en-v1.5`), normalised vectors |
| `vector_store.py` | FAISS inner-product index (= cosine similarity), saved to disk |
| `retriever.py` | Question -> embedding -> FAISS top-k |
| `generator.py` | Claude (if `ANTHROPIC_API_KEY` is set) or offline FLAN-T5 |
| `prompts.py` | Answer only from the excerpts; otherwise say the information is not in the documents |
| `pipeline.py` | Ties the steps together |

## Optional features (off by default)

Kept in the code but disabled so the core stays simple. Enable with environment variables:

| Variable | Feature | File |
|---|---|---|
| `RAG_RERANK=1` | Cross-encoder reranking of the top-20 FAISS results | `reranker.py` |
| `RAG_REWRITE=1` | Rewrite follow-up questions into standalone queries (Claude) | `pipeline.py` |
| `RAG_MEMORY=1` | Send chat history to the LLM (Claude) | `pipeline.py` |
| `RAG_STUDY_TOOLS=1` | Summaries, study questions, MCQs, document comparison (Claude) | `tools.py` |
| `RAG_ACCOUNTS=1` | Sign-up / sign-in; each user gets a private library and index under `users/data/<name>/` (passwords stored as salted scrypt hashes in `users/users.json`) | `rag/accounts.py`, `app.py` |
| `RAG_APP_PASSWORD=...` | Shared-password login screen (one password for everyone; no per-user accounts) | `app.py` |

## Evaluation on QASPER

```powershell
.\.venv\Scripts\python.exe prepare_qasper.py                      # 20 papers, 76 questions
.\.venv\Scripts\python.exe evaluate.py --dataset qasper           # Recall@k, Precision@k, MRR
.\.venv\Scripts\python.exe experiments.py --dataset qasper        # embedding / chunk-size / reranker comparison
.\.venv\Scripts\python.exe evaluate.py --dataset qasper --answers --limit 30   # grade answers (Claude, costs tokens)
```

QASPER annotators mark gold **evidence paragraphs**. A retrieved chunk counts as relevant if it covers at least half the sentences of a gold paragraph. Questions don't name their paper, so retrieval is reported both within the question's own paper (standard QASPER setting) and across all papers (harder).

## Results (QASPER subset: 20 papers, 72 answerable questions, retrieval scoped to the question's paper)

**Embedding model** (chunk size 256)

| Model | Recall@1 | Recall@5 | Recall@10 | MRR |
|---|---|---|---|---|
| all-MiniLM-L6-v2 | 0.29 | 0.68 | 0.86 | 0.457 |
| **bge-small-en-v1.5** (default) | **0.35** | 0.69 | 0.83 | **0.475** |
| all-mpnet-base-v2 | 0.29 | 0.68 | 0.88 | 0.458 |

**Chunk size** (tokens, overlap 40, bge-small)

| Size | Recall@1 | Recall@3 | Recall@5 | Recall@10 |
|---|---|---|---|---|
| 128 | 0.28 | 0.46 | 0.56 | 0.69 |
| 256 (default) | 0.35 | 0.56 | 0.69 | 0.83 |
| 384 | 0.28 | 0.58 | 0.69 | 0.96 |
| 500 | 0.32 | 0.51 | 0.72 | 0.94 |

Small chunks (128) hurt; larger chunks raise Recall@10 but dilute the text sent to the LLM.

**Cross-encoder reranking** (FAISS top-20, then rerank)

| Setup | Recall@3 | Recall@5 | Recall@10 | MRR | Latency |
|---|---|---|---|---|---|
| FAISS only | 0.56 | 0.69 | 0.83 | 0.475 | ~19 ms |
| + reranker | 0.67 | 0.79 | 0.92 | 0.516 | ~1.9 s |

Reranking improves Recall@5 by about 10 points and MRR by about 0.04, at roughly 100x the latency on CPU. It is why `RAG_RERANK=1` is worth trying but off by default.

Small sample (72 questions): differences of a few points are within noise.

## Limits

* Scanned PDFs have no text layer and are rejected (no OCR).
* Accounts are basic: no password reset, email verification or rate limiting, and the app should sit behind HTTPS if exposed publicly. Without `RAG_ACCOUNTS=1` everyone shares one library (optionally behind a single password).
* Retrieval is evaluated on a 20-paper subset of the QASPER validation split, not the whole dataset; figure/table evidence is excluded.
