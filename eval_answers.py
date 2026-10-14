"""
Answer-quality evaluation — the generation half of the eval harness.

`eval.py` measures RETRIEVAL (Hit@k, MRR). This measures the ANSWER the LLM
produces on top of retrieval — the part that actually reaches the user:

  gold-fact recall  : did the answer contain the known-correct fact?
  citation accuracy : did it cite the expected source document?
  groundedness      : is every claim supported by the retrieved context?
                      (LLM-as-judge, 0..1 — skipped with --no-judge)
  abstention        : on out-of-knowledge-base questions, did it correctly
                      refuse instead of hallucinating?

A pass/fail GATE (thresholds below) turns this into a prompt-regression test:
wire `python eval_answers.py` into CI and a prompt or retrieval change that
quietly degrades answers fails the build instead of shipping.

Run:  python eval_answers.py              # full run (uses the local LLM)
      python eval_answers.py --no-judge   # skip the groundedness judge
"""
from __future__ import annotations

import sys

from rag import retrieve, _format_context, _messages, _cite, get_llm
from eval import EVAL_SET  # reuse the labelled (question, gold, source) set

# Out-of-KB questions: the ONLY correct behaviour is to abstain. This is the
# cheapest, most honest hallucination guard — no judge needed.
ABSTAIN_SET = [
    "What is the capital of France?",
    "How do I reset my email password?",
    "Who won the 2018 football world cup?",
]

# Pass/fail gate. Tune as the system improves; a regression drops below these.
THRESHOLDS = {
    "gold_fact_recall": 0.70,
    "citation_accuracy": 0.60,
    "groundedness": 0.70,   # only enforced when the judge runs
    "abstention": 1.00,     # must abstain on every out-of-KB question
}

# Phrases that count as a correct refusal (the system prompt's exact line plus
# common paraphrases a local model may produce).
_ABSTAIN_MARKERS = (
    "don't have that in the provided documents",
    "do not have that in the provided documents",
    "not in the provided documents",
    "don't have",
    "not available in the",
    "no information",
)

_JUDGE_SYSTEM = (
    "You are a strict grader. Given CONTEXT and an ANSWER, decide whether every "
    "factual claim in the ANSWER is supported by the CONTEXT. Reply with ONLY a "
    "number from 0.0 to 1.0 — 1.0 = fully supported, 0.0 = unsupported/contradicted."
)


def _generate(question: str) -> tuple[str, str, list[str]]:
    """Retrieve, then generate. Returns (answer, context, cited_sources)."""
    items = retrieve(question)
    context = _format_context(items)
    answer = get_llm().invoke(_messages(question, context)).content
    sources = sorted({_cite(d) for d, _ in items})
    return answer, context, sources


def _judge_groundedness(context: str, answer: str) -> float:
    """LLM-as-judge faithfulness score in [0, 1]; robust to noisy output."""
    reply = get_llm().invoke([
        ("system", _JUDGE_SYSTEM),
        ("human", f"CONTEXT:\n{context}\n\nANSWER:\n{answer}\n\nScore:"),
    ]).content
    import re
    m = re.search(r"(\d(?:\.\d+)?)", reply)
    if not m:
        return 0.0
    return max(0.0, min(1.0, float(m.group(1))))


def _is_abstention(answer: str) -> bool:
    low = answer.lower()
    return any(marker in low for marker in _ABSTAIN_MARKERS)


def main() -> None:
    use_judge = "--no-judge" not in sys.argv

    print(f"Answer-quality eval — {len(EVAL_SET)} in-KB + {len(ABSTAIN_SET)} abstention "
          f"questions (judge={'on' if use_judge else 'off'})\n")

    gold_hits = cite_hits = 0
    ground_scores: list[float] = []

    print(f"{'#':>2}  {'gold':>4}  {'cite':>4}  {'ground':>6}  question")
    print("-" * 64)
    for i, (question, gold, expected_src) in enumerate(EVAL_SET, start=1):
        answer, context, _sources = _generate(question)
        low = answer.lower()

        gold_ok = gold.lower() in low
        cite_ok = expected_src.lower() in low
        gold_hits += gold_ok
        cite_hits += cite_ok

        ground = _judge_groundedness(context, answer) if use_judge else None
        if ground is not None:
            ground_scores.append(ground)

        gs = f"{ground:>6.2f}" if ground is not None else "   — "
        print(f"{i:>2}  {'✓' if gold_ok else '·':>4}  {'✓' if cite_ok else '·':>4}  {gs}  {question[:34]}")

    abstain_ok = 0
    for question in ABSTAIN_SET:
        answer, _c, _s = _generate(question)
        ok = _is_abstention(answer)
        abstain_ok += ok
        print(f" A  {'✓' if ok else '✗':>4}  {'—':>4}  {'   — '}  [abstain] {question[:30]}")

    n = len(EVAL_SET)
    results = {
        "gold_fact_recall": gold_hits / n,
        "citation_accuracy": cite_hits / n,
        "abstention": abstain_ok / len(ABSTAIN_SET),
    }
    if use_judge:
        results["groundedness"] = sum(ground_scores) / len(ground_scores) if ground_scores else 0.0

    print("\nScores vs gate:")
    print(f"{'metric':<20}{'score':>8}{'gate':>8}   result")
    print("-" * 46)
    failed = []
    for metric, score in results.items():
        gate = THRESHOLDS[metric]
        passed = score >= gate
        if not passed:
            failed.append(metric)
        print(f"{metric:<20}{score:>8.2f}{gate:>8.2f}   {'PASS' if passed else 'FAIL'}")

    if failed:
        print(f"\n❌ GATE FAILED: {', '.join(failed)}")
        sys.exit(1)
    print("\n✅ GATE PASSED")


if __name__ == "__main__":
    main()
