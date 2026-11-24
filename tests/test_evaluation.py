import math

from rag.evaluation import (answer_f1, answer_recall, bootstrap_ci, citation_check, evaluate_answers,
                            evaluate_retrieval, is_abstention, is_relevant, normalize_answer, token_f1)
from rag.prompts import NOT_FOUND


def test_normalize_and_token_f1():
    assert normalize_answer("The  BERT model!") == "bert model"
    assert token_f1("BERT model", "the BERT model") == 1.0
    assert token_f1("cats", "dogs") == 0.0
    # precision 2/4, recall 2/2 -> F1 = 2 * 0.5 * 1 / 1.5
    assert math.isclose(token_f1("w x y z", "w x"), 2 * 0.5 * 1.0 / 1.5)
    assert token_f1("", "") == 1.0


def test_answer_f1_uses_best_reference_and_ignores_citations():
    q = {"evidence": ["x"], "answer": "BERT", "answers": ["BERT", "a BERT based model"]}
    assert answer_f1("BERT [1]", q) == 1.0
    assert answer_f1(NOT_FOUND, q) == 0.0


def test_answer_recall_ignores_extra_words():
    q = {"evidence": ["x"], "answer": "SimpleQuestions; WebQSP", "answers": ["SimpleQuestions; WebQSP"]}
    wordy = "They achieve the state of the art on both SimpleQuestions and WebQSP benchmarks [1]."
    assert answer_recall(wordy, q) == 1.0          # the full answer is in there...
    assert answer_f1(wordy, q) < 0.4               # ...but F1 penalises the extra words
    assert answer_recall(NOT_FOUND, q) == 0.0
    assert answer_recall(NOT_FOUND, {"evidence": [], "answer": ""}) == 1.0


def test_answer_f1_unanswerable_rewards_abstaining():
    q = {"evidence": [], "answer": ""}
    assert answer_f1(NOT_FOUND, q) == 1.0
    assert answer_f1("It is 42.", q) == 0.0


def test_is_abstention():
    assert is_abstention(NOT_FOUND)
    assert is_abstention("The context does not contain this.")
    assert not is_abstention("Transformers use attention [1].")


def test_citation_check():
    assert citation_check("A [1][2]. B [7].", n_hits=3) == {"n_citations": 3, "invalid_citations": 1}
    assert citation_check("No cites.", 3) == {"n_citations": 0, "invalid_citations": 0}


def test_is_relevant_sentence_coverage():
    gold = "First sentence here. Second sentence here."
    assert is_relevant("blah. First sentence here. More.", [gold])           # 1 of 2 = 50%
    assert not is_relevant("Nothing relevant.", [gold])
    assert is_relevant("x LABELED data y", ["labeled data"])                  # short phrases


def test_bootstrap_ci_brackets_mean():
    values = [0.0, 1.0] * 50
    lo, hi = bootstrap_ci(values)
    assert lo < 0.5 < hi and 0.35 < lo and hi < 0.65
    assert bootstrap_ci([1.0] * 10) == (1.0, 1.0)
    assert all(math.isnan(x) for x in bootstrap_ci([]))


DOCS = {
    "pets.md": "# Cats\nCats are small furry pets that purr.\n\n# Dogs\nDogs are loyal pets that bark.\n",
    "nlp.md": "# Attention\nTransformers use self attention over tokens.\n",
}


def test_evaluate_retrieval_end_to_end(make_pipeline):
    p = make_pipeline(DOCS)
    p.sync_directory()
    qa = [
        {"question": "which pets bark", "answer": "dogs", "evidence": ["Dogs are loyal pets that bark."]},
        {"question": "what do transformers use", "answer": "attention",
         "evidence": ["Transformers use self attention over tokens."]},
        {"question": "unanswerable", "answer": "", "evidence": []},
    ]
    m = evaluate_retrieval(p, qa, ks=(1, 3), rerank=False, ci=True)
    assert m["n_questions"] == 2
    assert m["recall@3"] == 1.0
    assert "mrr_ci" in m and m["mrr_ci"][0] <= m["mrr"] <= m["mrr_ci"][1]
    m_h = evaluate_retrieval(p, qa, ks=(1, 3), rerank=False, hybrid=True)
    assert m_h["recall@3"] == 1.0


def test_evaluate_answers_without_judge(make_pipeline):
    p = make_pipeline(DOCS)
    p.sync_directory()
    qa = [{"question": "which pets bark", "answer": "dogs", "evidence": ["Dogs are loyal pets that bark."]},
          {"question": "unanswerable", "answer": "", "evidence": []}]
    res = evaluate_answers(p, qa)
    assert res["n_questions"] == 2
    assert len(res["rows"]) == 2
    assert res["answers_with_citations"] == 1.0          # FakeLLM always cites [1]
    assert res["invalid_citation_rate"] == 0.0
    assert res["abstain_rate_on_unanswerable"] == 0.0    # FakeLLM never abstains
    assert "correctness" not in res                      # no judge -> no LLM grades
