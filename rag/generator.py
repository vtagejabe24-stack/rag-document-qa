"""Phase 7 - LLM generation. Two backends with the same interface:
  * ClaudeLLM - Claude via the Anthropic API (best quality; needed for study tools)
  * LocalLLM  - FLAN-T5 running locally with PyTorch (offline, QA only)"""
from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path

from . import prompts

FALLBACK_BETA = "server-side-fallback-2026-07-01"


class LLMUnavailable(RuntimeError):
    pass


class ClaudeLLM:
    supports_tools = True

    def __init__(self, model: str, effort: str = "medium"):
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = model
        self.effort = effort
        self.name = f"Claude ({model})"

    def _stream(self, system: str, messages: list[dict], max_tokens: int, output_config: dict):
        # Server-side fallback: if a safety classifier declines, the API re-runs
        # the request on a fallback model inside the same call.
        return self.client.beta.messages.stream(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
            output_config=output_config,
            betas=[FALLBACK_BETA],
            fallbacks="default",
        )

    def stream(self, system: str, messages: list[dict], max_tokens: int = 8000,
               effort: str | None = None) -> Iterator[str]:
        with self._stream(system, messages, max_tokens, {"effort": effort or self.effort}) as s:
            yield from s.text_stream
            final = s.get_final_message()
        if final.stop_reason == "refusal":
            yield "\n\n_The model declined to answer this request._"
        elif final.stop_reason == "max_tokens":
            yield "\n\n_(response truncated at the max_tokens limit)_"

    def complete(self, system: str, messages: list[dict], max_tokens: int = 8000,
                 effort: str | None = None) -> str:
        return "".join(self.stream(system, messages, max_tokens, effort))

    def complete_json(self, system: str, messages: list[dict], schema: dict,
                      max_tokens: int = 16000) -> dict:
        output_config = {"effort": self.effort, "format": {"type": "json_schema", "schema": schema}}
        with self._stream(system, messages, max_tokens, output_config) as s:
            final = s.get_final_message()
        if final.stop_reason != "end_turn":
            raise RuntimeError(f"Generation stopped early (stop_reason={final.stop_reason}).")
        return json.loads(next(b.text for b in final.content if b.type == "text"))

    def answer_stream(self, question: str, context: str, history: list[dict]) -> Iterator[str]:
        messages = [{"role": m["role"], "content": m["content"]} for m in history]
        messages.append({
            "role": "user",
            "content": f"<documents>\n{context}\n</documents>\n\nQuestion: {question}",
        })
        return self.stream(prompts.ANSWER_SYSTEM, messages, max_tokens=4000)


class LocalLLM:
    """Offline generator running on the CPU with PyTorch. Two kinds of model work:
      * instruction-tuned chat models (e.g. Qwen2.5-Instruct): long context, follow the
        same cite-your-sources prompt as Claude, answers streamed token by token;
      * seq2seq models (e.g. FLAN-T5): small and fast, but a 512-token input limit.
    No conversation memory or study tools (those need Claude)."""

    supports_tools = False
    # Tokens we allow into the model. 2048 fits all 5 default excerpts (~1,450 tokens)
    # for chat models while keeping CPU prompt processing fast.
    INPUT_LIMIT = {"seq2seq": 512, "causal": 2048}
    MAX_NEW_TOKENS = 256

    def __init__(self, model_name: str):
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.kind = "seq2seq" if AutoConfig.from_pretrained(model_name).is_encoder_decoder else "causal"
        cls = AutoModelForSeq2SeqLM if self.kind == "seq2seq" else AutoModelForCausalLM
        # Force float32: recent transformers versions load a checkpoint's own dtype (often
        # bfloat16), which most CPUs have no fast instructions for. On a laptop CPU this
        # made Qwen2.5-0.5B read a 1,500-token prompt in ~1,100 s instead of ~9 s.
        self.model = cls.from_pretrained(model_name, dtype=torch.float32).eval()
        self.name = f"{model_name} (local)"

    # ---- context budget: the pipeline sends only as many excerpts as fit ----
    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def context_budget(self, question: str) -> int:
        overhead = self.count_tokens(self._prompt(question, "")) + 16
        if self.kind == "causal":
            overhead += self.count_tokens(prompts.ANSWER_SYSTEM)
        return max(64, self.INPUT_LIMIT[self.kind] - overhead)

    @staticmethod
    def _prompt(question: str, context: str) -> str:
        return (
            f"Answer the question using only the context. If the context does not contain "
            f"the answer, reply \"{prompts.NOT_FOUND}\"\n\nQuestion: {question}\n\nContext: {context}"
        )

    def answer_stream(self, question: str, context: str, history: list[dict]) -> Iterator[str]:
        if self.kind == "seq2seq":
            # Question goes first so any truncation cuts context, not the question.
            inputs = self.tokenizer(self._prompt(question, context), return_tensors="pt",
                                    truncation=True, max_length=self.INPUT_LIMIT["seq2seq"])
            with self.torch.no_grad():
                out = self.model.generate(**inputs, max_new_tokens=200, num_beams=4)
            yield self.tokenizer.decode(out[0], skip_special_tokens=True)
            return
        yield from self._chat_stream(question, context)

    def _chat_stream(self, question: str, context: str) -> Iterator[str]:
        from threading import Thread

        from transformers import TextIteratorStreamer

        messages = [
            {"role": "system", "content": prompts.ANSWER_SYSTEM},
            {"role": "user", "content": f"<documents>\n{context}\n</documents>\n\nQuestion: {question}"},
        ]
        ids = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True,
                                                 return_tensors="pt", return_dict=True)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        kwargs = dict(**ids, streamer=streamer, max_new_tokens=self.MAX_NEW_TOKENS, do_sample=False,
                      repetition_penalty=1.05)

        def run():
            with self.torch.no_grad():
                self.model.generate(**kwargs)

        Thread(target=run, daemon=True).start()
        yield from streamer

    def _unavailable(self, *args, **kwargs):
        raise LLMUnavailable("This feature needs Claude. Set ANTHROPIC_API_KEY and restart the app.")

    stream = complete = complete_json = _unavailable


def claude_configured() -> bool:
    return bool(
        os.environ.get("ANTHROPIC_API_KEY")
        or os.environ.get("ANTHROPIC_AUTH_TOKEN")
        or os.environ.get("ANTHROPIC_PROFILE")
        or (Path.home() / ".config" / "anthropic").exists()   # `ant auth login` profile
    )


def resolve_backend(backend: str) -> str:
    if backend == "auto":
        return "claude" if claude_configured() else "local"
    return backend


def make_llm(backend: str, claude_model: str, claude_effort: str, local_model: str):
    if resolve_backend(backend) == "claude":
        return ClaudeLLM(claude_model, claude_effort)
    return LocalLLM(local_model)
