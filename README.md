# RAG Knowledge Assistant

A small, real **Retrieval-Augmented Generation (RAG)** app: ask questions in natural
language and get answers grounded in your own documents, with source citations.

Built to learn the GenAI stack — **LangChain + FAISS + embeddings + a local LLM** — on top of
a plain **FastAPI** backend. Goes beyond a toy: **hybrid retrieval (dense + BM25) with
Reciprocal Rank Fusion and a cross-encoder re-ranker**, an **evaluation harness** for both
retrieval *and* answer quality, and an **agentic variant** (LangGraph). **Fully open-source and
runs locally** — embeddings, vector store, re-ranker, *and* the LLM (via Ollama). No API keys,
no cost, works offline, and no data leaves the machine.

> Personal learning project. The sample documents in `data/` are a short knowledge base about
> RAG engineering itself — drop your own `.md` / `.txt` / `.pdf` files in to index anything.

## How it works

```
  documents (data/)
        │  1. load + Markdown-header-aware chunking   (section metadata kept)
        ▼
     chunks ──► 2. embed (all-MiniLM-L6-v2, local) ──► FAISS index (.index/)
                                                          │  + incremental manifest
  question                                                │
     │                                                    ▼
     ├──► dense retrieval (MMR, diverse) ─┐
     │                                    ├─► 3. Reciprocal Rank Fusion (RRF)
     └──► BM25 keyword retrieval ─────────┘            │
                                                       ▼
                            4. cross-encoder re-rank ─► top-k chunks
                                                       │
                                                       ▼
                     5. LLM answers using ONLY those chunks, citing source › section
```

Grounding the model in retrieved chunks — and telling it to say *"I don't have that"* when
the answer isn't present — is what makes RAG reduce hallucination.

### Why hybrid + re-ranking?

- **Dense** retrieval captures meaning/paraphrase but is weak on exact terms (names, numbers,
  acronyms). **BM25** nails exact keywords but is blind to synonyms. **Hybrid** gets both.
- **RRF** fuses the two ranked lists by *rank*, so their incompatible score scales don't matter.
- A **cross-encoder re-ranker** then reads (query, chunk) together and reorders the shortlist —
  far more accurate than the first-stage embeddings, but only run on the top candidates.

## Setup

**1. Install Ollama** (the local open-source LLM runtime) from https://ollama.com, then pull a model:

```bash
ollama pull llama3.2        # ~2GB, small and fast; runs on CPU
```

**2. Install the Python app** (Python 3.11 recommended for the ML wheels):

```bash
cd localrag
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env         # defaults to Ollama — no key needed
```

First run downloads the local embedding model (~80MB) and the cross-encoder re-ranker (~80MB)
once. If model downloads hang, keep `HF_HUB_DISABLE_XET=1` in `.env` (some networks block the
newer Xet CDN). Want a cloud LLM instead? Set `LLM_PROVIDER=anthropic` (or `openai`) in `.env`,
add the key, and uncomment that package in `requirements.txt`.

## Use it

```bash
# 1) Build (or incrementally update) the index from ./data
python rag.py ingest

# 2) Ask from the command line
python rag.py ask "What is Reciprocal Rank Fusion and what constant does it use?"

# 3) Or run the web UI + API
uvicorn rag:app --reload
#   open http://localhost:8000          (chat page — streams answers, upload docs)
#   POST http://localhost:8000/ask         { "question": "..." }
#   POST http://localhost:8000/ask/stream  (streamed text/plain)
#   POST http://localhost:8000/ingest      (multipart file upload → indexes just that file)
```

`ingest` is **incremental**: only new or content-changed files are re-embedded (tracked by a
SHA-256 manifest in `.index/`), and deleted files are pruned automatically.

## Evaluate the retrieval

`eval.py` runs a labelled question set and reports **Hit@k** (correct chunk in the top-k) and
**MRR** (1 / rank of the first correct chunk), comparing configurations so every change is
justified by a before/after number:

```bash
python eval.py        # at k = TOP_K
python eval.py 1      # at k = 1 (most discriminative)
```

Measured on the sample corpus (11 questions):

| Config             | Hit@1 | MRR@4 |
|--------------------|:-----:|:-----:|
| dense (baseline)   | 0.82  | 0.886 |
| hybrid (no rerank) | 0.91  | 0.955 |
| **hybrid + rerank**| **1.00** | **1.000** |

Hybrid fusion lifts ranking over pure dense search, and the cross-encoder re-ranker puts the
correct chunk **first for every question**.

## Evaluate the answers (`eval_answers.py`)

`eval.py` scores *retrieval*; `eval_answers.py` scores the **generated answer** — the part the
user actually reads — and turns it into a **pass/fail regression gate** you can wire into CI:

| Metric | What it checks |
|--------|----------------|
| **gold-fact recall** | did the answer contain the known-correct fact? |
| **citation accuracy** | did it cite the expected source document? |
| **groundedness** | is every claim supported by the retrieved context? (**LLM-as-judge**, 0–1; skip with `--no-judge`) |
| **abstention** | on out-of-knowledge-base questions, did it correctly **refuse** instead of hallucinating? |

```bash
python eval_answers.py              # full run (uses the local LLM + judge)
python eval_answers.py --no-judge   # skip the groundedness judge (faster)
```

Each metric has a threshold (e.g. abstention must be **1.00** — refuse on *every* out-of-KB
question); a prompt or retrieval change that quietly degrades answers fails the gate instead of
shipping. This is the difference between *claiming* the RAG works and *proving* it still does.

## Agentic RAG (`agent.py`)

Plain RAG retrieves once and answers. The **agent** version instead *decides* when (and how
many times) to search — it reasons, calls tools, reads the results, and can call again before
answering. Built with **LangGraph** (`langchain.agents.create_agent`) over two tools:
`search_documents` (the full hybrid + re-rank pipeline) and `list_sources`.

```bash
python agent.py "List your documents, then explain how a cross-encoder re-ranker works."
```

## Retrieval knobs (`.env`)

| Var | Default | Effect |
|-----|---------|--------|
| `RETRIEVAL_MODE` | `hybrid` | `hybrid` (dense+MMR + BM25 + RRF) or `dense` baseline |
| `RERANK` | `true` | cross-encoder re-rank the fused candidates |
| `TOP_K` | `4` | chunks fed to the LLM |
| `DENSE_CANDIDATES` / `BM25_CANDIDATES` | `20` | pool size per retriever before fusion |
| `FUSE_TOP` | `12` | fused candidates sent to the re-ranker |
| `MMR_LAMBDA` | `0.5` | dense diversity (0 = max diverse, 1 = max relevant) |
| `RRF_K` | `60` | RRF smoothing constant |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `800` / `120` | chunking granularity |

## Stack

| Piece | Choice | Why |
|-------|--------|-----|
| Framework | LangChain | Standard GenAI toolkit |
| Vector store | FAISS | Open-source, fast, local, no external service |
| Embeddings | sentence-transformers (MiniLM, 384-d) | Open-source, local, no API key |
| Keyword retrieval | rank-bm25 (BM25Okapi) | Exact-term recall for hybrid search |
| Fusion | Reciprocal Rank Fusion | Score-scale-agnostic list merge |
| Re-ranker | cross-encoder `ms-marco-MiniLM-L-6-v2` | Accurate second-stage ordering |
| LLM | Ollama (Llama 3.2) — local | Free, offline, private; pluggable to Claude/GPT via `.env` |
| API | FastAPI | Lightweight, streaming + uploads |

**Everything is open-source and self-hosted** — the whole pipeline runs offline with zero
third-party API calls, so no document data ever leaves the box.

## What I learned

- The full RAG pipeline end-to-end: chunk → embed → store → **hybrid retrieve → fuse → re-rank**
  → generate, and how each stage moves the metrics.
- Why dense-only retrieval misses exact terms, and how BM25 + RRF + a cross-encoder fix it.
- How to **measure** retrieval quality (Hit@k, MRR) and justify changes with before/after numbers.
- How to **evaluate the generation half** too — gold-fact recall, citation accuracy,
  groundedness (LLM-as-judge) and abstention — and gate it so regressions can't ship.
- How structure-aware (header) chunking enables **section-precise citations**.
- Agentic vs plain RAG: letting the model *decide* when to retrieve (LangGraph tools) vs a
  fixed retrieve-then-answer pipeline.

## Possible next steps

- Swap FAISS for **pgvector** (Postgres) to practice a SQL-backed vector store.
- Query rewriting / multi-query expansion for higher recall on hard questions.
- Wire `eval_answers.py` into CI so every change runs the answer-quality gate.
