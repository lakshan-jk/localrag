"""
Retrieval evaluation — measure the improvement, don't just claim it.

For each labelled question we know a gold substring that only appears in the
correct chunk. We retrieve with several configurations and compute:

  Hit@k : fraction of questions whose correct chunk is in the top-k
  MRR   : mean of 1 / (rank of the first correct chunk)   (0 if not found)

We compare the plain dense baseline against hybrid and hybrid+re-rank, so every
retrieval change can be justified with a before/after number.

Run:  python eval.py            # default k = TOP_K
      python eval.py 5          # override k
"""
from __future__ import annotations

import sys

from rag import retrieve, TOP_K

# (question, gold substring that marks the correct chunk, expected source file)
EVAL_SET = [
    ("What is Reciprocal Rank Fusion and what constant does it use?",
     "k + rank", "retrieval.md"),
    ("How does a cross-encoder re-ranker read the query and candidate?",
     "one transformer", "retrieval.md"),
    ("What does BM25 score chunks with?",
     "inverse-document-frequency", "retrieval.md"),
    ("What keeps the top-k from being near-duplicate chunks?",
     "Maximal Marginal Relevance", "retrieval.md"),
    ("How many dimensions does all-MiniLM-L6-v2 output?",
     "384", "embeddings-and-chunking.md"),
    ("What is the downside of larger chunks?",
     "dilute the embedding", "embeddings-and-chunking.md"),
    ("Where does pgvector store its vectors?",
     "pgvector", "embeddings-and-chunking.md"),
    ("What is another name for Hit@k?",
     "recall@k", "generation-and-eval.md"),
    ("What does Mean Reciprocal Rank reward?",
     "Mean Reciprocal Rank", "generation-and-eval.md"),
    ("What is the lost in the middle effect?",
     "lost in the middle", "generation-and-eval.md"),
    ("What two prompt rules reduce hallucination?",
     "does not know", "generation-and-eval.md"),
]

CONFIGS = [
    ("dense (baseline)", dict(mode="dense")),
    ("hybrid (no rerank)", dict(mode="hybrid", rerank=False)),
    ("hybrid + rerank", dict(mode="hybrid", rerank=True)),
]


def first_relevant_rank(items, gold: str) -> int:
    """1-based rank of the first chunk containing the gold substring, else 0."""
    for i, (doc, _score) in enumerate(items, start=1):
        if gold.lower() in doc.page_content.lower():
            return i
    return 0


def evaluate(config: dict, k: int) -> tuple[float, float]:
    hits, rr = 0, 0.0
    for question, gold, _src in EVAL_SET:
        items = retrieve(question, k=k, **config)
        rank = first_relevant_rank(items, gold)
        if rank:
            hits += 1
            rr += 1.0 / rank
    n = len(EVAL_SET)
    return hits / n, rr / n


def main() -> None:
    k = int(sys.argv[1]) if len(sys.argv) > 1 else TOP_K
    print(f"Evaluating {len(EVAL_SET)} questions at k={k}\n")
    print(f"{'config':<22}{'Hit@k':>8}{'MRR':>8}")
    print("-" * 38)
    baseline = None
    for name, config in CONFIGS:
        hit, mrr = evaluate(config, k)
        if baseline is None:
            baseline = (hit, mrr)
        dh = (hit - baseline[0]) * 100
        print(f"{name:<22}{hit:>8.2f}{mrr:>8.3f}" + (f"   ({dh:+.0f}pt Hit@k)" if name != CONFIGS[0][0] else ""))
    print("\n(Hit@k = correct chunk in top-k; MRR = 1/rank of first correct chunk.)")


if __name__ == "__main__":
    main()
