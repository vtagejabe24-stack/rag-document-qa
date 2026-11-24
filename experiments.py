"""Phase 16 - model comparison experiments (retrieval only, no API key needed).

  python experiments.py --dataset qasper             # all three experiments
  python experiments.py --dataset qasper embeddings  # or: chunking | rerank

Results are printed as tables and saved to results/<dataset>_<experiment>.json."""
from __future__ import annotations

import argparse
import json
import time

import config
from rag.evaluation import evaluate_retrieval, load_eval
from rag.pipeline import RAGPipeline

KS = (1, 3, 5, 10)

ap = argparse.ArgumentParser()
ap.add_argument("which", nargs="?", default="all", choices=["all", "embeddings", "chunking", "rerank"])
ap.add_argument("--dataset", default="sample", choices=list(config.DATASETS))
ap.add_argument("--all-documents", action="store_true",
                help="search every document instead of only the question's own paper")
args = ap.parse_args()
SCOPED = not args.all_documents     # no effect on datasets whose questions have no "paper" field
DOCS_DIR, EVAL_FILE, _ = config.DATASETS[args.dataset]


def build(embed_model: str, chunk_size: int) -> tuple[RAGPipeline, float]:
    """An in-memory pipeline (nothing is persisted) built from the dataset's documents."""
    t0 = time.perf_counter()
    p = RAGPipeline(index_dir=None, docs_dir=DOCS_DIR, embed_model=embed_model,
                    chunk_size=chunk_size, llm_backend="local")
    p.sync_directory()
    return p, time.perf_counter() - t0


def table(title: str, rows: list[dict], label: str) -> None:
    print(f"\n=== {title} [{args.dataset}] ===")
    cols = [f"recall@{k}" for k in KS] + ["precision@5", "mrr", "latency_ms"]
    print(f"{label:<44}" + "".join(f"{c:>12}" for c in cols) + f"{'chunks':>9}{'index_s':>9}")
    for r in rows:
        print(f"{r[label]:<44}" + "".join(f"{r[c]:>12.3f}" for c in cols)
              + f"{r['n_chunks']:>9}{r['index_s']:>9.1f}")


def save(name: str, rows: list[dict]) -> None:
    config.RESULTS_DIR.mkdir(exist_ok=True)
    (config.RESULTS_DIR / f"{args.dataset}_{name}.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")


def exp_embeddings(qa) -> None:
    rows = []
    for model in config.EXPERIMENT_EMBED_MODELS:
        p, secs = build(model, config.CHUNK_SIZE)
        m = evaluate_retrieval(p, qa, KS, rerank=False, scoped=SCOPED)
        rows.append({"model": model, **m, "n_chunks": len(p.store.chunks), "index_s": secs})
    table("Embedding models (no reranker)", rows, "model")
    save("embeddings", rows)


def exp_chunking(qa) -> None:
    rows = []
    for size in config.EXPERIMENT_CHUNK_SIZES:
        p, secs = build(config.EMBED_MODEL, size)
        m = evaluate_retrieval(p, qa, KS, rerank=False, scoped=SCOPED)
        rows.append({"chunk_size": f"{size} tokens (overlap {p.chunk_overlap})", **m,
                     "n_chunks": len(p.store.chunks), "index_s": secs})
    table("Chunk size (no reranker)", rows, "chunk_size")
    save("chunking", rows)


def exp_rerank(qa) -> None:
    p, secs = build(config.EMBED_MODEL, config.CHUNK_SIZE)
    rows = []
    for name, flag in [("FAISS only", False), (f"FAISS top-{config.TOP_K_RETRIEVE} + cross-encoder", True)]:
        m = evaluate_retrieval(p, qa, KS, rerank=flag, scoped=SCOPED)
        rows.append({"setup": name, **m, "n_chunks": len(p.store.chunks), "index_s": secs})
    table("Reranking", rows, "setup")
    save("rerank", rows)


qa = load_eval(EVAL_FILE)
print(f"{sum(1 for q in qa if q['evidence'])} answerable questions from {EVAL_FILE}")
if args.which in ("all", "embeddings"):
    exp_embeddings(qa)
if args.which in ("all", "chunking"):
    exp_chunking(qa)
if args.which in ("all", "rerank"):
    exp_rerank(qa)

