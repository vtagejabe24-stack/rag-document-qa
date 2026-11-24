"""Phase 14 - study tools: summarization, question generation, MCQs, comparison.
These read whole documents (not retrieved chunks), so they need Claude's long context."""
from __future__ import annotations

from collections.abc import Iterator

from . import prompts
from .generator import LLMUnavailable

MCQ_SCHEMA = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "question": {"type": "string"},
                    "options": {"type": "array", "items": {"type": "string"}},
                    "answer_index": {"type": "integer"},
                    "explanation": {"type": "string"},
                    "location": {"type": "string"},
                },
                "required": ["question", "options", "answer_index", "explanation", "location"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["questions"],
    "additionalProperties": False,
}


def _require(llm) -> None:
    if not getattr(llm, "supports_tools", False):
        raise LLMUnavailable("Study tools need Claude. Set ANTHROPIC_API_KEY and restart the app.")


def _doc(name: str, text: str) -> str:
    return f'<document name="{name}">\n{text}\n</document>'


def summarize(llm, name: str, text: str) -> Iterator[str]:
    _require(llm)
    msg = f"{_doc(name, text)}\n\nSummarize this document."
    return llm.stream(prompts.SUMMARY_SYSTEM, [{"role": "user", "content": msg}], max_tokens=8000)


def generate_questions(llm, name: str, text: str, n: int) -> Iterator[str]:
    _require(llm)
    msg = f"{_doc(name, text)}\n\nWrite {n} study questions about this document."
    return llm.stream(prompts.QUESTIONS_SYSTEM, [{"role": "user", "content": msg}], max_tokens=8000)


def generate_mcqs(llm, name: str, text: str, n: int) -> list[dict]:
    _require(llm)
    msg = f"{_doc(name, text)}\n\nWrite {n} multiple-choice questions about this document."
    data = llm.complete_json(prompts.MCQ_SYSTEM, [{"role": "user", "content": msg}], MCQ_SCHEMA)
    # keep only well-formed items
    return [q for q in data["questions"] if len(q["options"]) == 4 and 0 <= q["answer_index"] < 4]


def compare_documents(llm, docs: dict[str, str], focus: str) -> Iterator[str]:
    _require(llm)
    blocks = "\n\n".join(_doc(name, text) for name, text in docs.items())
    msg = f"{blocks}\n\nCompare these documents. Focus: {focus}"
    return llm.stream(prompts.COMPARE_SYSTEM, [{"role": "user", "content": msg}], max_tokens=10000)
