# Embeddings and Chunking

## Embedding models

An embedding model maps text to a dense vector so that similar meanings land
near each other. `sentence-transformers/all-MiniLM-L6-v2` is a small, fast,
open-source model that outputs **384-dimensional** vectors and runs comfortably
on CPU. Larger models (e.g. `bge-large`) are more accurate but slower and need
more memory. Embeddings should be produced by the **same model** at ingest time
and at query time, otherwise the vectors are not comparable.

## Bi-encoders vs cross-encoders

A bi-encoder embeds the query and each document separately, so document vectors
can be precomputed and searched quickly — this is what powers first-stage
retrieval. A cross-encoder instead processes the query and document jointly and
is more accurate but cannot be precomputed, so it is reserved for re-ranking a
short candidate list.

## Chunking strategy

Documents are split into chunks because embeddings represent a bounded amount of
text and retrieval should return focused passages. Chunk size trades off:
**smaller chunks** are more precise but can fragment an idea across pieces;
**larger chunks** carry more context but dilute the embedding and add noise. A
typical starting point is 500–1000 characters with a 10–20% **overlap** so a
sentence split across a boundary still appears whole in one chunk.

## Structure-aware chunking

For Markdown, splitting on headers keeps each chunk within one section and lets
you attach the heading path as metadata. Carrying the section into the chunk's
metadata enables **section-precise citations** — you can tell the user not just
which file but which heading an answer came from.

## Vector stores

A vector store indexes embeddings for fast nearest-neighbour search. **FAISS** is
an open-source library that runs in-process and persists to local files, ideal
for a single machine. **pgvector** stores vectors inside PostgreSQL, which is
convenient when you already run Postgres and want SQL filtering. Hosted options
like Pinecone or Qdrant Cloud manage scaling for you at the cost of an external
dependency.
