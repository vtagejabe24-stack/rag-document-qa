"""Recompute answer metrics for saved answer-evaluation files, without regenerating
any answers (e.g. after a new metric is added to rag/evaluation.py).

  python rescore_answers.py results/answers_qasper_*.json --dataset qasper
"""
import argparse
import glob
import json

import config
from rag.evaluation import answer_f1, answer_recall, bootstrap_ci, is_abstention, load_eval

ap = argparse.ArgumentParser()
ap.add_argument("files", nargs="+")
ap.add_argument("--dataset", default="qasper", choices=list(config.DATASETS))
args = ap.parse_args()

qa = {q["question"]: q for q in load_eval(config.DATASETS[args.dataset][1])}
for path in [p for pattern in args.files for p in glob.glob(pattern)]:
    res = json.load(open(path, encoding="utf-8"))
    rows = res["rows"]
    for r in rows:
        q = qa[r["question"]]
        r["answer_f1"] = answer_f1(r["answer"], q)
        r["answer_recall"] = answer_recall(r["answer"], q)
        r["answer_words"] = len(r["answer"].split())
        r["abstained"] = is_abstention(r["answer"])

    def mean(xs):
        xs = list(xs)
        return sum(xs) / len(xs) if xs else float("nan")

    answerable = [r for r in rows if r["answerable"]]
    res.update({
        "answer_f1": mean(r["answer_f1"] for r in rows),
        "answer_f1_ci": list(bootstrap_ci([r["answer_f1"] for r in rows])),
        "answer_recall": mean(r["answer_recall"] for r in rows),
        "answer_recall_ci": list(bootstrap_ci([r["answer_recall"] for r in rows])),
        "answer_recall_answerable": mean(r["answer_recall"] for r in answerable),
        "median_answer_words": sorted(r["answer_words"] for r in rows)[len(rows) // 2],
    })
    res["rows"] = rows
    with open(path, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(f"{path}: Answer-F1 {res['answer_f1']:.3f}  answer recall {res['answer_recall']:.3f} "
          f"[{res['answer_recall_ci'][0]:.2f}, {res['answer_recall_ci'][1]:.2f}]  "
          f"median words {res['median_answer_words']}")
