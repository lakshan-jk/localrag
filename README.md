# RAG Knowledge Assistant

A small, real **Retrieval-Augmented Generation (RAG)** app: ask questions in natural
language and get answers grounded in your own documents, with source citations.

Built to learn the GenAI stack — **LangChain + FAISS + embeddings + a local LLM** — on top of
a plain **FastAPI** backend. **Fully open-source and runs locally**: embeddings, vector store,
*and* the LLM (via Ollama). No API keys, no cost, works offline, and no data leaves the machine.

> Personal learning project. The sample documents describe the YourBackers OTT platform;
> drop your own `.md` / `.txt` / `.pdf` files into `data/` to index anything.

## How it works

```
  documents (data/)
        │  1. load + split into chunks   (RecursiveCharacterTextSplitter)
        ▼
     chunks
        │  2. embed each chunk           (all-MiniLM-L6-v2, runs locally)
        ▼
   vectors ──► 3. store in FAISS index   (.index/)
                         │
  question ─► embed ─► 4. similarity search ─► top-k relevant chunks
                                                     │
                                                     ▼
                              5. LLM answers using ONLY those chunks,
                                 citing source filenames   (Claude / GPT)
```

Grounding the model in retrieved chunks — and telling it to say *"I don't have that"* when
the answer isn't present — is what makes RAG reduce hallucination.

## Setup

**1. Install Ollama** (the local open-source LLM runtime) from https://ollama.com, then pull a model:

```bash
ollama pull llama3.2        # ~2GB, small and fast; runs on CPU
```

**2. Install the Python app:**

```bash
cd rag-knowledge-assistant
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env         # defaults to Ollama — no key needed
```

First run also downloads the local embedding model (~80MB) once.
Want to use a cloud model instead? Set `LLM_PROVIDER=anthropic` (or `openai`) in `.env`,
add the key, and uncomment that package in `requirements.txt`.

## Use it

```bash
# 1) Build the index from ./data
python rag.py ingest

# 2) Ask from the command line
python rag.py ask "How is video content protected from piracy?"

# 3) Or run the web UI + API
uvicorn rag:app --reload
#   open http://localhost:8000      (tiny chat page)
#   POST http://localhost:8000/ask  { "question": "..." }
```

## Agentic RAG (`agent.py`)

Plain RAG retrieves once and answers. The **agent** version instead *decides* when (and how
many times) to search — it reasons, calls tools, reads the results, and can call again before
answering. Built with **LangGraph** (`langchain.agents.create_agent`) over two tools:
`search_documents` (semantic search on the FAISS index) and `list_sources`.

```bash
python agent.py "List the documents, then explain how content is protected from piracy."
```

Plain RAG vs. agent: use plain RAG when one retrieval answers the question; use an agent when
the task needs multiple lookups, tool use, or iteration.

## Things to try (and talk about in an interview)

- Change `TOP_K` in `.env` and see how more/less context changes answers.
- Change `CHUNK_SIZE` / `CHUNK_OVERLAP` and re-`ingest` — smaller chunks = more precise but
  more fragmented; larger = more context but noisier retrieval.
- Ask something *not* in the docs and confirm it honestly says it doesn't know.
- Swap `LLM_PROVIDER` between `anthropic` and `openai`.

## Stack

| Piece | Choice | Why |
|-------|--------|-----|
| Framework | LangChain | Standard GenAI toolkit; wanted hands-on experience |
| Vector store | FAISS | Open-source, fast, local, no external service |
| Embeddings | sentence-transformers (MiniLM) | Open-source, runs locally, no API key |
| LLM | Ollama (Llama 3.2) — open-source, local | Free, offline, private; pluggable to Claude/GPT via `.env` |
| API | FastAPI | Lightweight Python web framework |

**Everything is open-source and self-hosted** — a nice property to be able to explain: the whole
RAG pipeline runs offline with zero third-party API calls, so no document data ever leaves the box.

## What I learned

- How a RAG pipeline fits together end-to-end: chunk → embed → store → retrieve → generate.
- Why chunking strategy and `top-k` directly affect answer quality.
- How a grounding system prompt + citations reduces hallucination.
- Where this would need to grow for production: a managed vector DB (e.g. pgvector/Pinecone),
  re-ranking, evaluation (RAGAS), caching, auth, and observability.

## Possible next steps

- Swap FAISS for **pgvector** (Postgres) to practice a SQL-backed vector store.
- Add a `/ingest` upload endpoint so files can be added via the API.
- Add retrieval evaluation and a re-ranking step.
