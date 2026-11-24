"""Evaluate the full system.

  python evaluate.py --dataset qasper                       # retrieval metrics (free, offline)
  python evaluate.py --dataset qasper --answers             # + Answer-F1 with the local LLM (free, offline)
  python evaluate.py --dataset qasper --answers --llm claude --judge --limit 30
                                                            # Claude answers, graded by Claude (costs tokens)
  python evaluate.py --dataset qasper_full --hybrid         # 890 questions, hybrid search
  python evaluate.py                                        # the small built-in sample set

Run `python prepare_qasper.py` (or `--full`) first for the QASPER datasets.
"""
import argparse
import json

import config
from rag.evaluation import evaluate_answers, evaluate_retrieval, load_eval
from rag.generator import ClaudeLLM
from rag.pipeline import RAGPipeline

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", default="sample", choices=list(config.DATASETS))
ap.add_argument("--answers", action="store_true", help="generate answers and score them (Answer-F1)")
ap.add_argument("--llm", default="auto", choices=["auto", "claude", "local"],
                help="which LLM answers (auto = Claude if configured, else the local model)")
ap.add_argument("--local-model", default=config.LOCAL_LLM, help="Hugging Face id of the local LLM")
ap.add_argument("--judge", action="store_true", help="also grade answers with Claude (LLM-as-judge)")
ap.add_argument("--hybrid", action="store_true", default=config.ENABLE_HYBRID,
                help="BM25 + FAISS hybrid search for the answers")
ap.add_argument("--rerank", action="store_true", default=config.ENABLE_RERANKER,
                help="cross-encoder reranking for the answers")
ap.add_argument("--limit", type=int, help="only answer N questions (spread evenly across the set)")
ap.add_argument("--skip-retrieval", action="store_true")
args = ap.parse_args()

config.LOCAL_LLM = args.local_model
docs_dir, eval_file, index_dir = config.DATASETS[args.dataset]
qa = load_eval(eval_file)
pipe = RAGPipeline(index_dir=index_dir, docs_dir=docs_dir, llm_backend=args.llm)
report = pipe.sync_directory()
print(f"Dataset '{args.dataset}': {len(report)} documents, {len(pipe.store.chunks)} chunks, "
      f"{len(qa)} questions ({sum(1 for q in qa if q['evidence'])} answerable)")

has_papers = any("paper" in q for q in qa)
if not args.skip_retrieval:
    settings = [("search all documents", False), ("search the question's paper", True)] if has_papers \
        else [("search all documents", False)]
    print("\n--- Retrieval (95% bootstrap CI in brackets) ---")
    for scope_name, scoped in settings:
        for name, kw in [("dense", {}), ("hybrid", {"hybrid": True}),
                         ("dense + reranker", {"rerank": True}), ("hybrid + reranker", {"hybrid": True, "rerank": True})]:
            m = evaluate_retrieval(pipe, qa, rerank=kw.get("rerank", False), hybrid=kw.get("hybrid", False),
                                   scoped=scoped, ci=True)
            print(f"[{scope_name:<27}] {name:>17}: "
                  f"R@1={m['recall@1']:.3f}  R@5={m['recall@5']:.3f} [{m['recall@5_ci'][0]:.2f}-{m['recall@5_ci'][1]:.2f}]  "
                  f"R@10={m['recall@10']:.3f}  MRR={m['mrr']:.3f} [{m['mrr_ci'][0]:.2f}-{m['mrr_ci'][1]:.2f}]  "
                  f"{m['latency_ms']:.0f} ms")

if args.answers:
    subset = qa
    if args.limit and args.limit < len(qa):
        step = len(qa) / args.limit
        subset = [qa[int(i * step)] for i in range(args.limit)]
    judge = None
    if args.judge:
        judge = pipe.llm if pipe.supports_tools else ClaudeLLM(config.CLAUDE_MODEL, config.CLAUDE_EFFORT)
    print(f"\n--- Answers on {len(subset)} questions from {pipe.llm_name} "
          f"({'hybrid' if args.hybrid else 'dense'}{' + reranker' if args.rerank else ''}) ---")
    res = evaluate_answers(pipe, subset, scoped=has_papers, judge=judge, rerank=args.rerank, hybrid=args.hybrid)
    for k, v in res.items():
        if k == "rows":
            continue
        print(f"{k:>34}: " + (f"[{v[0]:.3f}, {v[1]:.3f}]" if isinstance(v, list) else f"{v:.3f}"))
    config.RESULTS_DIR.mkdir(exist_ok=True)
    tag = "claude" if pipe.supports_tools else config.LOCAL_LLM.split("/")[-1]
    out = config.RESULTS_DIR / f"answers_{args.dataset}_{tag}.json"
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\nSummary and per-question results saved to {out}")
