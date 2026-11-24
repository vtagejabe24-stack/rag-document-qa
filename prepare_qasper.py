"""Build a QASPER evaluation corpus.

QASPER (Dasigi et al., 2021) = questions about NLP research papers, written by
readers of the title+abstract and answered by annotators with gold *evidence
paragraphs* from the full text. Each paper becomes one Markdown document (sections
become headings, so citations show the section) and each question becomes one line
of data/qasper/qa.jsonl.

  python prepare_qasper.py                       # 20 papers from the validation split
  python prepare_qasper.py --full                 # whole validation split -> dataset "qasper_full"
  python prepare_qasper.py --split test --papers 50
"""
from __future__ import annotations

import argparse
import json
import re

import pandas as pd
from huggingface_hub import hf_hub_download

import config

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="validation", choices=["train", "validation", "test"])
ap.add_argument("--papers", type=int, default=20, help="0 = all papers in the split")
ap.add_argument("--full", action="store_true",
                help="every paper with at least one answerable question -> data/qasper_full")
ap.add_argument("--min-questions", type=int, default=3, help="skip papers with fewer answerable questions")
args = ap.parse_args()

path = hf_hub_download("allenai/qasper", f"qasper/{args.split}/0000.parquet", repo_type="dataset",
                       revision="refs/convert/parquet", local_dir=config.ROOT / "data" / "qasper_raw")
df = pd.read_parquet(path)

name = "qasper_full" if args.full else "qasper"
if args.full:
    args.papers, args.min_questions = 0, 1
docs_dir, eval_file, _ = config.DATASETS[name]


def clean(text: str | None) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def reference_answer(ans: dict) -> str:
    if ans["free_form_answer"].strip():
        return clean(ans["free_form_answer"])
    if ans["yes_no"] is not None and not pd.isna(ans["yes_no"]):
        return "Yes" if ans["yes_no"] else "No"
    return clean("; ".join(ans["extractive_spans"]))


def usable_evidence(ans: dict) -> list[str]:
    # figure/table evidence ("FLOAT SELECTED: ...") is not in the extracted text
    return [clean(e) for e in ans["evidence"] if e.strip() and not e.startswith("FLOAT SELECTED")]


docs_dir.mkdir(parents=True, exist_ok=True)
for old in docs_dir.glob("*.md"):      # clear files, not the folder (OneDrive can lock folders)
    old.unlink()

qa_rows, n_papers = [], 0
for _, paper in df.iterrows():
    rows = []
    for q, answers in zip(paper["qas"]["question"], paper["qas"]["answers"]):
        anns = list(answers["answer"])
        unans = [a["unanswerable"] for a in anns]
        if sum(unans) * 2 > len(anns):                       # majority say unanswerable
            rows.append({"question": clean(q), "answer": "", "evidence": []})
            continue
        ann = next((a for a in anns if not a["unanswerable"] and usable_evidence(a)), None)
        if ann is None:                                      # only figure/table evidence -> skip
            continue
        # every annotator's answer, for QASPER's max-over-annotators Answer-F1
        refs = list(dict.fromkeys(reference_answer(a) for a in anns if not a["unanswerable"]))
        rows.append({"question": clean(q), "answer": reference_answer(ann), "answers": refs,
                     "evidence": usable_evidence(ann)})
    if sum(1 for r in rows if r["evidence"]) < args.min_questions:
        continue

    name = f"{paper['id']}.md"
    ft = paper["full_text"]
    md = [f"# {clean(paper['title'])}", "", "## Abstract", "", clean(paper["abstract"]), ""]
    for section, paras in zip(ft["section_name"], ft["paragraphs"]):
        md += [f"## {clean(section) or 'Untitled'}", ""]
        md += [clean(p) + "\n" for p in paras if clean(p)]
    (docs_dir / name).write_text("\n".join(md), encoding="utf-8")
    qa_rows += [{**r, "paper": name} for r in rows]

    n_papers += 1
    if n_papers == args.papers:
        break

eval_file.parent.mkdir(parents=True, exist_ok=True)
with open(eval_file, "w", encoding="utf-8") as f:
    for r in qa_rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

answerable = sum(1 for r in qa_rows if r["evidence"])
print(f"{n_papers} papers -> {docs_dir}")
print(f"{len(qa_rows)} questions ({answerable} answerable, {len(qa_rows) - answerable} unanswerable) -> {eval_file}")
