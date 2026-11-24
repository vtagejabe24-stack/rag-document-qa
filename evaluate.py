"""Evaluate the full system.

  python evaluate.py --dataset qasper             # retrieval metrics on QASPER (free, offline)
  python evaluate.py --dataset qasper --answers --limit 30   # also grade answers with Claude (costs tokens)
  python evaluate.py                              # the small built-in sample set

Run `python prepare_qasper.py` first for the QASPER dataset.
"""
import argparse
import json

import config
from rag.evaluation import evaluate_answers, evaluate_retrieval, load_eval
from rag.pipeline import RAGPipeline

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="sample", choices=list(config.DATASETS))
ap.add_argument("--answers", action="store_true", help="grade generated answers (needs Claude)")
ap.add_argument("--limit", type=int, help="only grade N questions (spread evenly across the set)")
args = ap.parse_args()

docs_dir, eval_file, index_dir = config.DATASETS[args.dataset]
qa = load_eval(eval_file)
pipe = RAGPipeline(index_dir=index_dir, docs_dir=docs_dir, llm_backend="claude" if args.answers else "local")
report = pipe.sync_directory()
print(f"Dataset '{args.dataset}': {len(report)} documents, {len(pipe.store.chunks)} chunks, "
      f"{len(qa)} questions ({sum(1 for q in qa if q['evidence'])} answerable)")

has_papers = any("paper" in q for q in qa)
settings = [("search all documents", False), ("search the question's paper", True)] if has_papers \
    else [("search all documents", False)]
print("\n--- Retrieval ---")
for scope_name, scoped in settings:
    for name, flag in [("without reranker", False), ("with reranker", True)]:
        m = evaluate_retrieval(pipe, qa, rerank=flag, scoped=scoped)
        print(f"[{scope_name:<27}] {name:>16}: "
              + "  ".join(f"{k}={v:.3f}" for k, v in m.items() if k not in ("n_questions", "precision@1", "precision@10")))

if args.answers:
    subset = qa
    if args.limit and args.limit < len(qa):
        step = len(qa) / args.limit
        subset = [qa[int(i * step)] for i in range(args.limit)]
    print(f"\n--- Answers on {len(subset)} questions (LLM-as-judge, scores 1-5) ---")
    res = evaluate_answers(pipe, subset, scoped=has_papers)
    for k, v in res.items():
        if k != "rows":
            print(f"{k:>34}: {v:.3f}")
    config.RESULTS_DIR.mkdir(exist_ok=True)
    out = config.RESULTS_DIR / f"answers_{args.dataset}.json"
    out.write_text(json.dumps(res["rows"], indent=2), encoding="utf-8")
    print(f"\nPer-question grades saved to {out}")
