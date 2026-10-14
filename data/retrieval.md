# Retrieval Methods

## Dense retrieval

Dense retrieval encodes the query and every chunk into fixed-length embedding
vectors and finds the nearest chunks by cosine similarity or Euclidean (L2)
distance. It captures **semantic** similarity, so it can match paraphrases that
share no words. Its weakness is exact terms: rare product names, error codes,
and acronyms are often embedded poorly, so pure dense search can miss them.

## Sparse / BM25 retrieval

BM25 is a bag-of-words ranking function that scores chunks by term-frequency and
inverse-document-frequency. It excels at **exact keyword** matches — names,
identifiers, and rare tokens — but is blind to synonyms and paraphrase. BM25
needs no embedding model and is extremely fast.

## Hybrid retrieval

Hybrid retrieval runs dense and BM25 side by side and combines their results, so
the system gets semantic recall from vectors and exact-match precision from
keywords. It is the single most reliable retrieval-quality upgrade over pure
dense search.

## Reciprocal Rank Fusion

Reciprocal Rank Fusion (RRF) merges several ranked lists without needing to
normalize their scores. Each document gets a fused score of the sum over lists of
`1 / (k + rank)`, where `rank` is its position in that list and `k` is a constant
(commonly 60). Because it uses ranks, not raw scores, RRF is robust when the
sub-retrievers produce scores on totally different scales — exactly the dense vs.
BM25 case.

## Cross-encoder re-ranking

A first-stage retriever returns many candidates cheaply. A cross-encoder
re-ranker then reads the query and each candidate **together** in one transformer
pass and outputs a relevance score. This is far more accurate than the
independent (bi-encoder) embeddings used for first-stage search, but too slow to
run over the whole corpus — so it is applied only to the top candidates. A common
open model is `cross-encoder/ms-marco-MiniLM-L-6-v2`.

## Maximal Marginal Relevance

Maximal Marginal Relevance (MMR) selects results that are both relevant to the
query and diverse from each other, controlled by a lambda between 0 and 1. It
prevents the top-k from being several near-duplicate chunks of the same passage,
which wastes the context window.
