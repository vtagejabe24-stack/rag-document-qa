"""Streamlit web app:  streamlit run app.py"""
from __future__ import annotations

import hmac
import os
import re
import tempfile
from pathlib import Path

import streamlit as st

import config
from rag.generator import LLMUnavailable
from rag.highlight import best_sentences, highlight_html
from rag.loader import SUPPORTED
from rag.pipeline import RAGPipeline
from rag import accounts, tools

st.set_page_config(page_title="AI Document Assistant", page_icon="📄", layout="wide")


def require_login() -> None:
    """Optional shared-password gate: set RAG_APP_PASSWORD to enable. Off by default."""
    password = os.environ.get("RAG_APP_PASSWORD", "")
    if not password or st.session_state.get("authed"):
        return
    st.title("🔒 AI Document Assistant")
    with st.form("login"):
        entered = st.text_input("Password", type="password")
        if st.form_submit_button("Sign in"):
            if hmac.compare_digest(entered.encode(), password.encode()):
                st.session_state.authed = True
                st.rerun()
            st.error("Wrong password.")
    st.stop()


def require_account() -> str | None:
    """Optional per-user accounts (RAG_ACCOUNTS=1). Returns the signed-in username."""
    if not config.ENABLE_ACCOUNTS:
        return None
    if st.session_state.get("user"):
        return st.session_state.user
    st.title("🔒 AI Document Assistant")
    signin, signup = st.tabs(["Sign in", "Create account"])
    with signin, st.form("signin"):
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        if st.form_submit_button("Sign in"):
            if accounts.verify(u, p):
                st.session_state.user = u.strip().lower()
                st.session_state.messages = []
                st.rerun()
            st.error("Wrong username or password.")
    with signup, st.form("signup"):
        u = st.text_input("Choose a username")
        p = st.text_input("Choose a password", type="password")
        if st.form_submit_button("Create account"):
            err = accounts.register(u, p)
            if err:
                st.error(err)
            else:
                st.session_state.user = u.strip().lower()
                st.session_state.messages = []
                st.rerun()
    st.stop()


if config.ENABLE_ACCOUNTS:
    user = require_account()
else:
    user = None
    require_login()


@st.cache_resource(show_spinner="Loading models and index...")
def get_pipeline(user: str | None) -> RAGPipeline:
    if user:                                   # private library per account
        docs_dir, index_dir = accounts.user_dirs(user)
        p = RAGPipeline(index_dir=index_dir, docs_dir=docs_dir)
    else:
        p = RAGPipeline()
    p.sync_directory()
    return p


pipe = get_pipeline(user)
st.session_state.setdefault("messages", [])

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("📄 Documents")
    if user:
        st.caption(f"Signed in as **{user}**")
        if st.button("Sign out"):
            st.session_state.pop("user", None)
            st.session_state.messages = []
            st.rerun()
    uploads = st.file_uploader(
        "Upload documents", type=[e.lstrip(".") for e in sorted(SUPPORTED)], accept_multiple_files=True
    )
    if uploads and st.button("Index uploaded files", type="primary"):
        pipe.docs_dir.mkdir(parents=True, exist_ok=True)
        for up in uploads:
            if up.size > config.MAX_UPLOAD_MB * 1024 * 1024:
                st.error(f"{up.name}: larger than {config.MAX_UPLOAD_MB} MB")
                continue
            name = Path(up.name).name                       # strip any path components
            dest = pipe.docs_dir / name
            dest.write_bytes(up.getbuffer())
            try:
                with st.spinner(f"Indexing {name}..."):
                    status = pipe.add_file(dest)
            except Exception as e:                          # corrupt / encrypted files
                dest.unlink(missing_ok=True)
                st.error(f"{name}: could not be read ({e})")
                continue
            if status == "empty":
                dest.unlink(missing_ok=True)
                st.warning(f"{name}: no text found, even with OCR")
            else:
                st.success(f"{name}: {status}")
        st.rerun()

    docs = pipe.documents()
    st.subheader(f"Library ({len(docs)})")
    for name, n in docs.items():
        c1, c2 = st.columns([5, 1])
        c1.markdown(f"✓ **{name}**  \n<small>{n} chunks</small>", unsafe_allow_html=True)
        if c2.button("🗑", key=f"del_{name}", help=f"Remove {name}"):
            pipe.remove_document(name, delete_file=True)
            st.rerun()

    selected = st.multiselect("Search only in", list(docs), placeholder="All documents")
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

    with st.expander("Advanced settings"):
        st.caption(f"LLM: {pipe.llm_name}" if pipe.supports_tools else f"LLM: {config.LOCAL_LLM} (offline, free)")
        st.caption(f"Embeddings: {config.EMBED_MODEL}")
        use_hybrid = st.toggle("Hybrid search (keywords + meaning)", value=config.ENABLE_HYBRID,
                               help="Adds BM25 keyword matching to the embedding search. Helps with exact "
                                    "names, acronyms and numbers.")
        use_rerank = st.toggle("Cross-encoder reranking (slower)", value=config.ENABLE_RERANKER,
                               help="A second model reorders the top passages more carefully. "
                                    "More accurate, about half a second slower per question.")
        top_k = st.slider("Chunks sent to the LLM", 1, 10, config.TOP_K_CONTEXT,
                          help="How many matching passages the answering model receives. "
                               "Small offline models only read as many as fit their input limit.")


# ------------------------------------------------------------------ helpers
def source_match(score: float) -> str:
    """Label for how relevant the best source is (cross-encoder score), not whether
    the answer is right. Cut points are near the quartiles of answerable QASPER questions."""
    level = "🟢 High" if score >= 2 else "🟡 Medium" if score >= -3 else "🔴 Low"
    return f"Source match: {level}"


def show_sources(sources) -> None:
    """sources: list of (excerpt number, hit, sentences to highlight)."""
    if not sources:
        return
    scores = [h.rerank_score for _, h, _ in sources if h.rerank_score is not None]
    if scores:                                  # only when reranking is on
        st.caption(source_match(max(scores)))
    with st.expander(f"Sources ({len(sources)})"):
        for i, h, marks in sources:
            rr = f" · rerank {h.rerank_score:.2f}" if h.rerank_score is not None else ""
            st.markdown(f"**[{i}] 📄 {h.chunk.source}** — 📍 {h.chunk.location()}  \n"
                        f"<small>similarity {h.score:.2f}{rr}</small>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size:0.9em;opacity:0.85'>{highlight_html(h.chunk.text, marks)}</div>",
                        unsafe_allow_html=True)


def cited(text: str, hits) -> list:
    """The excerpts the answer cites (all of them if it cites none), keeping their
    original numbers, each with the sentence that best supports the answer."""
    nums = {int(n) for n in re.findall(r"\[(\d+)\]", text)}
    picked = [(i, h) for i, h in enumerate(hits, 1) if i in nums] or list(enumerate(hits, 1))
    return [(i, h, best_sentences(pipe.embedder, text, h.chunk.text)) for i, h in picked]


def run_tool(label, fn, *args):
    try:
        with st.spinner(f"{label}..."):
            return fn(*args)
    except LLMUnavailable as e:
        st.error(str(e))
    except Exception as e:
        st.error(f"{label} failed: {e}")


# Example questions, offered only when the sample document that answers them is in the library.
EXAMPLES = {
    "deep_learning.md": ["What is the difference between a CNN and an RNN?",
                         "What problem do LSTM networks solve?",
                         "What is the capital of France?"],          # shows the "not found" reply
}


def chat_as_markdown(messages: list[dict]) -> str:
    parts = ["# Conversation"]
    for m in messages:
        parts.append(f"**{'You' if m['role'] == 'user' else 'Assistant'}:** {m['content']}")
        sources = [f"- [{i}] {h.chunk.source}, {h.chunk.location()}" for i, h, _ in m.get("hits") or []]
        if sources:
            parts.append("Sources:\n" + "\n".join(sources))
    return "\n\n".join(parts) + "\n"


# ------------------------------------------------------------------ tabs
if config.ENABLE_STUDY_TOOLS:
    chat_tab, tools_tab = st.tabs(["💬 Chat", "🎓 Study tools"])
else:
    chat_tab, tools_tab = st.container(), None

with chat_tab:
    st.header("AI Document Assistant")
    if not docs:
        st.info("Upload a document in the sidebar to get started.")
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("hits"):
                show_sources(m["hits"])

    clicked = None
    examples = [q for name, qs in EXAMPLES.items() if name in docs for q in qs]
    if examples:
        st.caption("Try an example:")
        for col, example in zip(st.columns(len(examples)), examples):
            if col.button(example, use_container_width=True):
                clicked = example

    typed = st.chat_input("Ask a question about your documents...", disabled=not docs)
    if question := typed or clicked:
        history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            try:
                with st.spinner("Searching..."):
                    ans = pipe.answer(question, history, selected or None, top_k, use_rerank, use_hybrid)
                if ans.search_query != question:
                    st.caption(f"🔎 Searched for: *{ans.search_query}*")
                text = st.write_stream(ans.stream)
                bad = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text)} - set(range(1, len(ans.hits) + 1)))
                if bad:
                    st.caption("⚠️ The answer cites " + ", ".join(f"[{n}]" for n in bad)
                               + ", which isn't one of the excerpts it was given. Treat those parts with care.")
                shown = cited(text, ans.hits) if "couldn't find" not in text else []
                show_sources(shown)
                st.session_state.messages.append({"role": "assistant", "content": text, "hits": shown})
            except Exception as e:
                st.error(f"Something went wrong: {e}")

    if st.session_state.messages:
        st.download_button("⬇ Download chat", chat_as_markdown(st.session_state.messages),
                           file_name="chat.md", mime="text/markdown")

with (tools_tab if tools_tab is not None else st.empty()):
    if tools_tab is None:
        pass                      # study tools are off (set RAG_STUDY_TOOLS=1 to enable)
    elif not pipe.supports_tools:
        st.warning("Study tools need Claude. Set ANTHROPIC_API_KEY and restart the app.")
    elif not docs:
        st.info("Upload a document first.")
    else:
        names = list(docs)
        doc = st.selectbox("Document", names)
        st.divider()
        s1, s2, s3, s4 = st.tabs(["Summary", "Questions", "MCQs", "Compare"])

        with s1:
            if st.button("Summarize document"):
                text = pipe.document_text(doc)
                st.write_stream(tools.summarize(pipe.llm, doc, text))

        with s2:
            n = st.number_input("How many questions", 3, 30, 10, key="nq")
            if st.button("Generate questions"):
                st.write_stream(tools.generate_questions(pipe.llm, doc, pipe.document_text(doc), int(n)))

        with s3:
            n = st.number_input("How many MCQs", 3, 20, 5, key="nm")
            if st.button("Generate MCQs"):
                st.session_state.mcqs = run_tool("Writing questions", tools.generate_mcqs,
                                                 pipe.llm, doc, pipe.document_text(doc), int(n))
            for i, q in enumerate(st.session_state.get("mcqs") or [], 1):
                st.markdown(f"**Q{i}. {q['question']}**")
                for letter, opt in zip("ABCD", q["options"]):
                    st.markdown(f"- {letter}. {opt}")
                with st.expander("Answer"):
                    st.markdown(f"**{'ABCD'[q['answer_index']]}** — {q['explanation']}"
                                + (f"  \n📍 {q['location']}" if q["location"] else ""))

        with s4:
            picked = st.multiselect("Documents to compare", names, default=names[:2])
            focus = st.text_input("Focus", "What are the main differences?")
            if st.button("Compare") and len(picked) >= 2:
                texts = {d: pipe.document_text(d) for d in picked}
                st.write_stream(tools.compare_documents(pipe.llm, texts, focus))
            elif len(picked) < 2:
                st.caption("Pick at least two documents.")
