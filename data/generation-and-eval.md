# Generation and Evaluation

## Grounding and hallucination

Retrieval-Augmented Generation reduces hallucination by forcing the model to
answer from retrieved context instead of parametric memory. Two prompt rules do
most of the work: tell the model to use **only** the provided context, and tell
it to explicitly say it does not know when the answer is absent. Adding citations
lets a reader verify each claim against its source.

## Context construction

The retrieved chunks are concatenated into the prompt, usually newest-to-oldest
by relevance, each labelled with its source so the model can cite it. If too many
chunks are stuffed in, the model can lose the relevant one in the middle — the
"lost in the middle" effect — so a re-ranker that puts the best chunk first
measurably improves answers.

## Retrieval metrics

Retrieval quality is measured separately from generation. **Hit@k** (also called
recall@k) is the fraction of questions whose correct source appears in the top k
retrieved chunks. **Mean Reciprocal Rank (MRR)** averages `1 / rank_of_first
_relevant_result` across questions, rewarding systems that place the right chunk
higher. Both need a small labelled set of questions paired with their expected
source.

## End-to-end evaluation

Beyond retrieval, answer quality can be scored with a framework such as **RAGAS**,
which estimates faithfulness (is every claim supported by the context?), answer
relevance, and context precision/recall. A cheap first check is keyword or
exact-match scoring against reference answers. Always evaluate retrieval and
generation as a **before/after** comparison so each change can be justified.

## Caching and cost

Caching embeddings and answers avoids recomputing work for repeated queries. For
local open-source stacks the main cost is latency, not dollars; for hosted LLMs,
caching and a smaller re-ranked context directly cut token spend.
