# RAG-Based Document Question Answering System

[![tests](https://github.com/vtagejabe24-stack/rag-document-qa/actions/workflows/tests.yml/badge.svg)](https://github.com/vtagejabe24-stack/rag-document-qa/actions/workflows/tests.yml)

Upload documents (PDF, DOCX, PPTX, TXT, MD, or scanned pages and images), ask questions, and get an answer **with the source text and page/section it came from**, with the supporting sentence highlighted. Built from the components up (no LangChain) and evaluated on the **QASPER** benchmark (questions about NLP research papers).

```
Document -> extract text (+ tables, + OCR for scans) -> chunk -> Sentence-BERT embeddings -> FAISS + BM25
Question -> embed -> hybrid search (FAISS + BM25, fused) -> optional cross-encoder rerank -> LLM -> answer + cited sources
```

## Key findings

Measured on the full QASPER validation split (276 papers, 890 answerable questions):

* **The cross-encoder reranker is the biggest win:** MRR 0.429 → 0.474 and Recall@1 0.273 → 0.316, both clearly beyond the 95% confidence intervals. It costs a few hundred ms per query on a CPU, so it's a toggle.
* **Hybrid search (BM25 + embeddings) helps a little on every metric** (Recall@10 0.839 → 0.866, MRR 0.429 → 0.440) for ~5 ms per query, so it is **on by default**. The gain is consistent but within the confidence intervals.
* **The reranker only needs to rescore the top 10 candidates:** MRR 0.474 with 10 vs 0.470 with 20, at lower latency.
* **Among offline models, Qwen2.5-1.5B finds the answer most often, but wordily.** Its answers contain the reference answer far more often (answer recall 0.61 vs 0.37 for FLAN-T5), but they are twice as long, so word-overlap Answer-F1 can't tell the models apart (0.22–0.26). FLAN-T5 is the default because it is ~6× faster.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:ANTHROPIC_API_KEY = "sk-ant-..."     # optional - without it a local model answers, fully offline
.\.venv\Scripts\python.exe -m streamlit run app.py
```

The app opens at http://localhost:8501 (add `--server.port 8502` if that port is busy). Sample documents in `data/documents/` are indexed on first launch. Embeddings are saved in `vector_db/`, so unchanged files are never re-embedded. Models are downloaded from Hugging Face on first use and cached, so later runs work offline.

### Tests

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest                  # 46 tests, ~1–2 min
.\.venv\Scripts\python.exe -m pytest -m "not slow"    # skip the OCR test
```

The tests use a tiny fake embedder, so they need no model downloads. They also run on GitHub Actions on every push (`.github/workflows/tests.yml`).

## Core modules (`rag/`)

| File | Role |
|---|---|
| `loader.py` | Text extraction with metadata (file, page/slide, section). PDF and Word tables are kept row by row (`Table: Model = BERT; F1 = 0.81.`); scanned PDF pages and images are read with OCR (RapidOCR, offline) |
| `chunker.py` | Sentence-aware chunks of ~256 tokens with overlap; each chunk knows its pages/sections |
| `embeddings.py` | Pretrained Sentence-BERT-style encoder (`BAAI/bge-small-en-v1.5`), normalised vectors |
| `vector_store.py` | FAISS inner-product index (= cosine similarity), saved to disk |
| `bm25.py` | Okapi BM25 keyword index and reciprocal rank fusion |
| `retriever.py` | Dense (FAISS) or hybrid (FAISS + BM25) search, optional cross-encoder rerank |
| `generator.py` | Claude (if `ANTHROPIC_API_KEY` is set) or an offline Hugging Face model (instruction-tuned chat models or FLAN-T5) |
| `pipeline.py` | Ties the steps together; sends the LLM only as many whole excerpts as fit its input window |
| `highlight.py` | Finds the sentence in each cited excerpt that best supports the answer |
| `prompts.py` | Answer only from the excerpts and cite them; otherwise say the information is not in the documents |
| `evaluation.py` | Retrieval metrics, QASPER Answer-F1, citation checks, bootstrap confidence intervals, optional LLM-as-judge |

## Options (environment variables)

| Variable | Feature | Default |
|---|---|---|
| `RAG_HYBRID=0` | Turn off BM25 keyword search fused with FAISS (also a toggle in the app) | on |
| `RAG_LOCAL_LLM=...` | Offline model, e.g. `Qwen/Qwen2.5-1.5B-Instruct` (finds answers more often, ~6× slower, streams its answer) | `google/flan-t5-base` |
| `RAG_RERANK=1` | Cross-encoder reranking of the top retrieved chunks (also a toggle) | off |
| `RAG_LLM=local` | Force the offline model even when Claude is configured (`claude` forces Claude) | auto |
| `RAG_OCR=0` | Turn off OCR for scanned PDFs and images | on |
| `RAG_REWRITE=1` | Rewrite follow-up questions into standalone queries (Claude) | off |
| `RAG_MEMORY=1` | Send chat history to the LLM (Claude) | off |
| `RAG_STUDY_TOOLS=1` | Summaries, study questions, MCQs, document comparison (Claude) | off |
| `RAG_ACCOUNTS=1` | Sign-up / sign-in; each user gets a private library and index under `users/data/<name>/` (passwords stored as salted scrypt hashes) | off |
| `RAG_APP_PASSWORD=...` | Shared-password login screen | off |

## Evaluation on QASPER

```powershell
.\.venv\Scripts\python.exe prepare_qasper.py            # 20-paper subset: 76 questions (quick)
.\.venv\Scripts\python.exe prepare_qasper.py --full     # whole validation split: 276 papers, 948 questions

.\.venv\Scripts\python.exe evaluate.py --dataset qasper_full                       # retrieval metrics with 95% CIs
.\.venv\Scripts\python.exe experiments.py --dataset qasper_full retrieval          # dense vs hybrid vs reranker
.\.venv\Scripts\python.exe experiments.py --dataset qasper embeddings              # or: chunking
.\.venv\Scripts\python.exe evaluate.py --dataset qasper --answers --llm local      # Answer-F1, free and offline
.\.venv\Scripts\python.exe evaluate.py --dataset qasper --answers --llm claude --judge --limit 30   # costs tokens
```

**How relevance is judged.** QASPER annotators mark gold **evidence paragraphs**. A retrieved chunk counts as relevant if it covers at least half the sentences of a gold paragraph. Questions don't name their paper, so retrieval is scored within the question's own paper (the standard QASPER setting). `evaluate.py` also reports the harder search across all papers.

**Answer metrics.** *Answer-F1* is QASPER's official metric: word overlap with the closest of the annotators' reference answers (abstaining is correct for unanswerable questions). It needs no API key. *Answer recall* (share of the reference's words found in the answer) is reported next to it because F1 penalises correct but wordy answers. `rescore_answers.py` recomputes both for saved results without regenerating answers. The evaluation also reports how often the model abstains correctly and wrongly, how many answers cite their sources, and whether citations point at real excerpts. With `--judge`, Claude additionally grades correctness, relevance, faithfulness and completeness from 1 to 5.

**Confidence intervals.** Every retrieval metric comes with a 95% bootstrap confidence interval (1,000 resamples over questions), so you can tell real improvements from noise.

## Results

### Retrieval strategies: full QASPER validation (890 answerable questions, search within the question's paper)

| Setup | Recall@1 | Recall@5 [95% CI] | Recall@10 | MRR [95% CI] |
|---|---|---|---|---|
| FAISS (dense) only | 0.273 | 0.645 [0.61–0.68] | 0.839 | 0.429 [0.40–0.45] |
| **Hybrid: BM25 + FAISS** (default) | 0.275 | 0.657 [0.63–0.69] | 0.866 | 0.440 [0.41–0.47] |
| FAISS top-10 + cross-encoder | **0.316** | 0.690 [0.66–0.72] | 0.839 | **0.474** [0.45–0.50] |
| FAISS top-20 + cross-encoder | 0.302 | 0.707 [0.68–0.74] | 0.873 | 0.470 [0.45–0.50] |
| Hybrid top-10 + cross-encoder | 0.307 | 0.704 [0.68–0.74] | 0.866 | 0.473 [0.45–0.50] |
| Hybrid top-20 + cross-encoder | 0.303 | **0.710** [0.68–0.74] | **0.879** | 0.473 [0.45–0.50] |

Dense and hybrid search take ~20–30 ms per query; reranking adds roughly 0.4–0.8 s on a CPU. With a top-10 rerank, Recall@10 can't change (the same 10 chunks are only reordered).

### Local answer models: QASPER 20-paper subset (30 questions, hybrid search)

| Model | Answer-F1 [95% CI] | Answer recall [95% CI] | Answers containing the reference (of 26 answerable) | Median answer length | Answers citing a source | Time per answer (CPU) |
|---|---|---|---|---|---|---|
| google/flan-t5-base (default) | 0.216 [0.10–0.36] | 0.371 [0.22–0.55] | 8 | 9 words | 10% | ~5 s |
| Qwen/Qwen2.5-0.5B-Instruct | 0.264 [0.14–0.40] | 0.383 [0.24–0.53] | 7 | 10 words | 3% (citation invalid) | ~16 s |
| Qwen/Qwen2.5-1.5B-Instruct | 0.230 [0.15–0.31] | **0.613** [0.46–0.76] | **15** | 18 words | 13% (all valid) | ~32 s |

*Answer recall* is the share of the reference answer's words that appear in the model's answer; "containing the reference" means recall ≥ 0.8. The two metrics pull in opposite directions. Answer-F1 penalises long answers: Qwen-1.5B's "They achieve the state of the art on both SimpleQuestions and WebQSP benchmarks" is correct but scores F1 0.31. Recall ignores extra words, so it rewards padding. Read together, they show Qwen-1.5B finds the answer far more often, but wordily. With 30 questions the intervals still overlap slightly, so treat this as a strong indication rather than proof. FLAN-T5 stays the default because it's ~6× faster. None of the small models reliably says "not found" on unanswerable questions.

Error breakdown (`results/answers_qasper_*.json`): FLAN-T5's failures are mostly retrieval-side. Its 512-token window holds about one excerpt, and for 14 of 26 answerable questions that excerpt didn't contain the evidence. Qwen-0.5B usually had the evidence but answered wrongly (14 of 26). Qwen-1.5B contained the reference in 15 of 26.

Local models are loaded in float32 on purpose. Loading Qwen in its stored bfloat16 format made it about 100× slower on a laptop CPU, which has no fast bfloat16 instructions.

### Embedding model and chunk size: QASPER 20-paper subset (72 answerable questions, dense search)

| Model (chunk size 256) | Recall@1 | Recall@5 | Recall@10 | MRR |
|---|---|---|---|---|
| all-MiniLM-L6-v2 | 0.29 | 0.68 | 0.86 | 0.457 |
| **bge-small-en-v1.5** (default) | **0.35** | 0.69 | 0.83 | **0.475** |
| all-mpnet-base-v2 | 0.29 | 0.68 | 0.88 | 0.458 |

| Chunk size (tokens, overlap 40) | Recall@1 | Recall@3 | Recall@5 | Recall@10 |
|---|---|---|---|---|
| 128 | 0.28 | 0.46 | 0.56 | 0.69 |
| 256 (default) | 0.35 | 0.56 | 0.69 | 0.83 |
| 384 | 0.28 | 0.58 | 0.69 | 0.96 |
| 500 | 0.32 | 0.51 | 0.72 | 0.94 |

Small chunks (128) hurt; larger chunks raise Recall@10 but dilute the text sent to the LLM. These two comparisons use only 72 questions, so differences of a few points are within noise.

## Deploying a live demo (Hugging Face Spaces, no Docker)

1. Create a new Space at huggingface.co/new-space and choose the **Streamlit** SDK.
2. Push this repository to the Space. Add this header to the top of the Space's `README.md`:
   ```yaml
   ---
   title: RAG Document QA
   sdk: streamlit
   app_file: app.py
   ---
   ```
3. Optionally add `ANTHROPIC_API_KEY` under the Space's **Settings → Secrets** for Claude answers. Without it the local model is used (the free CPU tier works, but answers are slow).
4. If the demo is public, set `RAG_APP_PASSWORD` or `RAG_ACCOUNTS=1` so strangers can't fill up your library.

## Limits

* PDF tables are extracted row by row only when they have full grid lines (common in business documents and Word exports). Research-paper tables drawn with horizontal rules only aren't detected as tables. Their text is still indexed as ordinary page text, just without the row structure.
* OCR quality depends on the scan; handwriting and very low-resolution images read poorly.
* Accounts are basic: no password reset, email verification or rate limiting, and the app should sit behind HTTPS if exposed publicly. Without `RAG_ACCOUNTS=1` everyone shares one library (optionally behind a single password).
* QASPER figure/table evidence is excluded from the benchmark (it isn't in the dataset's extracted text).
* Answer-F1 rewards short answers that reuse the reference wording, so a correct but wordy answer can score low. Answer recall has the opposite bias. Neither replaces human (or LLM-judge) grading.
