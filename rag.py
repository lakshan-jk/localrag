"""
RAG Knowledge Assistant — a small, real Retrieval-Augmented Generation app.

Pipeline:
  1. Load .md / .txt / .pdf files from ./data
  2. Chunk them (Markdown-header-aware) and embed with a local, free model
  3. Store vectors in a FAISS index (./.index) with an incremental manifest
  4. On a question:
       dense (MMR) + BM25 keyword  ->  Reciprocal Rank Fusion
       ->  cross-encoder re-rank  ->  top-k chunks
       ->  LLM answers using ONLY those chunks, citing source › section.

Run:
  python rag.py ingest              # build/update the index from ./data
  python rag.py ask "your question" # ask from the command line
  uvicorn rag:app --reload          # web API + chat UI -> http://localhost:8000
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

load_dotenv()

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
INDEX_DIR = os.getenv("INDEX_DIR", ".index")
MANIFEST_PATH = Path(INDEX_DIR) / "manifest.json"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))
TOP_K = int(os.getenv("TOP_K", "4"))
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")  # "ollama" = fully local & open-source

# --- Retrieval knobs (see .env.example) ---
RETRIEVAL_MODE = os.getenv("RETRIEVAL_MODE", "hybrid")   # "hybrid" | "dense"
RERANK = os.getenv("RERANK", "true").lower() == "true"
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
_min = os.getenv("RERANK_MIN_SCORE", "").strip()
RERANK_MIN_SCORE = float(_min) if _min else None
DENSE_CANDIDATES = int(os.getenv("DENSE_CANDIDATES", "20"))
BM25_CANDIDATES = int(os.getenv("BM25_CANDIDATES", "20"))
FUSE_TOP = int(os.getenv("FUSE_TOP", "12"))
MMR_LAMBDA = float(os.getenv("MMR_LAMBDA", "0.5"))
RRF_K = int(os.getenv("RRF_K", "60"))

# The system prompt is what keeps answers grounded: only use retrieved context,
# admit when the answer isn't there (this is how RAG reduces hallucination).
SYSTEM_PROMPT = (
    "You answer questions using ONLY the provided context.\n"
    'If the answer is not in the context, say: "I don\'t have that in the provided documents."\n'
    "Be concise. Cite the source you used in square brackets exactly as given after "
    "'[source: ...]', e.g. [retrieval.md › Reciprocal Rank Fusion]."
)

# --------------------------------------------------------------------------
# Cached singletons — models and the index are expensive to build repeatedly.
# --------------------------------------------------------------------------
_EMB = None
_LLM = None
_RERANKER = None
_STORE = None
_BM25 = None  # (BM25Okapi, [Document]) tuple, rebuilt when the index changes


def get_embeddings():
    """Local, free embedding model. Downloads once (~80MB) on first use."""
    global _EMB
    if _EMB is None:
        _EMB = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return _EMB


def get_llm():
    """The generation model. Local open-source (Ollama) by default; cloud optional."""
    global _LLM
    if _LLM is not None:
        return _LLM
    if LLM_PROVIDER == "anthropic":
        from langchain_anthropic import ChatAnthropic
        _LLM = ChatAnthropic(model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6"), temperature=0)
    elif LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        _LLM = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)
    else:
        from langchain_ollama import ChatOllama
        _LLM = ChatOllama(model=os.getenv("OLLAMA_MODEL", "llama3.2"), temperature=0)
    return _LLM


def get_reranker():
    """Cross-encoder that scores (query, chunk) jointly. Downloads once (~80MB)."""
    global _RERANKER
    if _RERANKER is None:
        from sentence_transformers import CrossEncoder
        _RERANKER = CrossEncoder(RERANK_MODEL)
    return _RERANKER


# --------------------------------------------------------------------------
# Loading + chunking
# --------------------------------------------------------------------------
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_MD_HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _read_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader
        return "\n".join((page.extract_text() or "") for page in PdfReader(str(path)).pages)
    return path.read_text(encoding="utf-8", errors="ignore")


def _iter_files() -> list[Path]:
    return [
        p for p in sorted(DATA_DIR.rglob("*"))
        if p.suffix.lower() in {".md", ".txt", ".pdf"} and p.is_file()
    ]


def split_file(path: Path) -> list[Document]:
    """Chunk one file into Documents with {source, section, cid} metadata.

    Markdown is split on headers first (so a chunk stays inside one section and
    carries its heading path), then long sections are sub-split by characters.
    """
    text = _read_text(path)
    if not text.strip():
        return []
    source = path.name
    char_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

    sections: list[tuple[str, str]] = []  # (section_path, content)
    if path.suffix.lower() == ".md":
        header_splitter = MarkdownHeaderTextSplitter(_MD_HEADERS, strip_headers=False)
        for sec in header_splitter.split_text(text):
            path_str = " › ".join(
                sec.metadata[key] for _, key in _MD_HEADERS if sec.metadata.get(key)
            )
            sections.append((path_str, sec.page_content))
    else:
        sections.append(("", text))

    chunks: list[Document] = []
    for section_path, content in sections:
        for piece in char_splitter.split_text(content):
            chunks.append(Document(
                page_content=piece,
                metadata={"source": source, "section": section_path},
            ))
    # Stable per-file chunk ids so incremental ingest can delete/replace cleanly.
    for i, c in enumerate(chunks):
        c.metadata["cid"] = f"{source}::{i}"
    return chunks


# --------------------------------------------------------------------------
# Index persistence + incremental ingest
# --------------------------------------------------------------------------
def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {"files": {}}


def _save_manifest(manifest: dict) -> None:
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2))


def load_index() -> FAISS | None:
    """Load (and cache) the FAISS index, or None if it hasn't been built."""
    global _STORE
    if _STORE is not None:
        return _STORE
    if not Path(INDEX_DIR, "index.faiss").exists():
        return None
    # allow_dangerous_deserialization is safe here: we built this index ourselves.
    _STORE = FAISS.load_local(INDEX_DIR, get_embeddings(), allow_dangerous_deserialization=True)
    return _STORE


def ingest(only: list[str] | None = None) -> dict:
    """Incrementally build/update the index.

    Only files whose content hash changed (or new files) are re-embedded; deleted
    files are pruned. Pass `only=[filename]` to ingest a single uploaded file.
    A missing manifest means a fresh build (any stale index is ignored).
    """
    global _STORE, _BM25
    manifest = _load_manifest()
    fresh = not MANIFEST_PATH.exists()
    store = None if fresh else load_index()

    current = {p.name: p for p in _iter_files()}
    targets = current if only is None else {n: current[n] for n in only if n in current}

    changed = [(n, p, _hash_file(p)) for n, p in targets.items()
               if manifest["files"].get(n, {}).get("hash") != _hash_file(p)]
    removed = [] if only is not None else [n for n in manifest["files"] if n not in current]

    # 1) Delete old vectors for changed + removed files.
    del_ids: list[str] = []
    for n, _, _ in changed:
        del_ids += manifest["files"].get(n, {}).get("ids", [])
    for n in removed:
        del_ids += manifest["files"][n].get("ids", [])
    if store is not None and del_ids:
        present = [i for i in del_ids if i in store.docstore._dict]
        if present:
            store.delete(present)

    # 2) Embed + add new/changed chunks.
    new_docs: list[Document] = []
    new_ids: list[str] = []
    per_file: dict[str, dict] = {}
    for n, p, h in changed:
        file_chunks = split_file(p)
        ids = [c.metadata["cid"] for c in file_chunks]
        new_docs += file_chunks
        new_ids += ids
        per_file[n] = {"hash": h, "ids": ids}

    if new_docs:
        if store is None:
            store = FAISS.from_documents(new_docs, get_embeddings(), ids=new_ids)
        else:
            store.add_documents(new_docs, ids=new_ids)

    for n in removed:
        manifest["files"].pop(n, None)
    manifest["files"].update(per_file)

    if store is not None:
        Path(INDEX_DIR).mkdir(parents=True, exist_ok=True)
        store.save_local(INDEX_DIR)
        _save_manifest(manifest)

    _STORE = store          # refresh caches
    _BM25 = None
    n_chunks = len(store.docstore._dict) if store is not None else 0
    return {
        "changed": [n for n, _, _ in changed],
        "removed": removed,
        "total_chunks": n_chunks,
        "total_files": len(manifest["files"]),
    }


# --------------------------------------------------------------------------
# Retrieval:  dense (MMR) + BM25  ->  RRF  ->  cross-encoder re-rank
# --------------------------------------------------------------------------
def _all_chunks(store: FAISS) -> list[Document]:
    return list(store.docstore._dict.values())


def _bm25(store: FAISS):
    """Build (and cache) a BM25 index over the current chunks."""
    global _BM25
    docs = _all_chunks(store)
    if _BM25 is None or len(_BM25[1]) != len(docs):
        from rank_bm25 import BM25Okapi
        _BM25 = (BM25Okapi([_tokenize(d.page_content) for d in docs]), docs)
    return _BM25


def _dense_candidates(store: FAISS, query: str, n: int) -> list[Document]:
    # MMR diversifies candidates so we don't fuse several near-duplicate chunks.
    return store.max_marginal_relevance_search(
        query, k=min(n, len(store.docstore._dict)), fetch_k=n * 3, lambda_mult=MMR_LAMBDA
    )


def _bm25_candidates(store: FAISS, query: str, n: int) -> list[Document]:
    bm, docs = _bm25(store)
    scores = bm.get_scores(_tokenize(query))
    order = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)[:n]
    return [docs[i] for i in order]


def _rrf(ranked_lists: list[list[Document]], k: int) -> list[Document]:
    """Reciprocal Rank Fusion: merge ranked lists by rank, not by raw score."""
    fused: dict[str, float] = {}
    by_cid: dict[str, Document] = {}
    for lst in ranked_lists:
        for rank, d in enumerate(lst):
            cid = d.metadata.get("cid") or hashlib.sha1(d.page_content.encode()).hexdigest()
            by_cid[cid] = d
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (k + rank + 1)
    return [by_cid[c] for c in sorted(fused, key=fused.get, reverse=True)]


def _rerank(query: str, docs: list[Document], top_k: int) -> list[tuple[Document, float]]:
    if not docs:
        return []
    scores = get_reranker().predict([(query, d.page_content) for d in docs])
    ranked = sorted(zip(docs, (float(s) for s in scores)), key=lambda x: x[1], reverse=True)
    if RERANK_MIN_SCORE is not None:
        kept = [(d, s) for d, s in ranked if s >= RERANK_MIN_SCORE]
        ranked = kept or ranked[:1]  # never return nothing if there were candidates
    return ranked[:top_k]


def retrieve(query: str, mode: str | None = None, k: int | None = None,
             rerank: bool | None = None) -> list[tuple[Document, float | None]]:
    """Return top-k (Document, score) for a query. `mode` = 'hybrid' | 'dense'."""
    store = load_index()
    if store is None:
        return []
    mode = mode or RETRIEVAL_MODE
    k = k or TOP_K
    rerank = RERANK if rerank is None else rerank

    if mode == "dense":
        return [(d, None) for d in store.similarity_search(query, k=k)]

    dense = _dense_candidates(store, query, DENSE_CANDIDATES)
    sparse = _bm25_candidates(store, query, BM25_CANDIDATES)
    fused = _rrf([dense, sparse], RRF_K)[:FUSE_TOP]
    if rerank:
        return _rerank(query, fused, k)
    return [(d, None) for d in fused[:k]]


# --------------------------------------------------------------------------
# Answer generation
# --------------------------------------------------------------------------
def _cite(d: Document) -> str:
    src = d.metadata.get("source", "?")
    sec = d.metadata.get("section")
    return f"{src} › {sec}" if sec else src


def _format_context(items: list[tuple[Document, float | None]]) -> str:
    return "\n\n---\n\n".join(f"[source: {_cite(d)}]\n{d.page_content}" for d, _ in items)


def _messages(question: str, context: str):
    return [("system", SYSTEM_PROMPT), ("human", f"Context:\n{context}\n\nQuestion: {question}")]


def ask(question: str) -> dict:
    """Retrieve, then have the LLM answer from the retrieved chunks with citations."""
    items = retrieve(question)
    if not items:
        return {"answer": "No index found. Run:  python rag.py ingest", "sources": []}
    reply = get_llm().invoke(_messages(question, _format_context(items)))
    sources = sorted({_cite(d) for d, _ in items})
    return {"answer": reply.content, "sources": sources}


def answer_stream(question: str):
    """Yield the answer token-by-token, then a final Sources line."""
    items = retrieve(question)
    if not items:
        yield "No index found. Run:  python rag.py ingest"
        return
    for chunk in get_llm().stream(_messages(question, _format_context(items))):
        if chunk.content:
            yield chunk.content
    yield "\n\nSources: " + ", ".join(sorted({_cite(d) for d, _ in items}))


# --------------------------------------------------------------------------
# Web API — run with:  uvicorn rag:app --reload
# --------------------------------------------------------------------------
from fastapi import FastAPI, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

app = FastAPI(title="RAG Knowledge Assistant")


class Question(BaseModel):
    question: str


@app.get("/health")
def health():
    store = load_index()
    return {
        "status": "ok",
        "indexed": store is not None,
        "chunks": len(store.docstore._dict) if store is not None else 0,
        "provider": LLM_PROVIDER,
        "retrieval_mode": RETRIEVAL_MODE,
        "rerank": RERANK,
    }


@app.post("/ask")
def ask_endpoint(q: Question):
    return ask(q.question)


@app.post("/ask/stream")
def ask_stream_endpoint(q: Question):
    return StreamingResponse(answer_stream(q.question), media_type="text/plain")


@app.post("/ingest")
async def ingest_endpoint(file: UploadFile):
    """Upload a .md/.txt/.pdf, save it to DATA_DIR, and index just that file."""
    name = Path(file.filename or "upload.txt").name
    if Path(name).suffix.lower() not in {".md", ".txt", ".pdf"}:
        return {"error": "Only .md, .txt, .pdf files are supported."}
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / name).write_bytes(await file.read())
    return ingest(only=[name])


PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>RAG Assistant</title>
<style>
 body{font-family:system-ui,sans-serif;max-width:680px;margin:48px auto;padding:0 16px;color:#1a1a1a}
 h2{margin-bottom:4px} p.sub{color:#666;margin-top:0}
 input[type=text]{width:100%;padding:11px;font-size:15px;border:1px solid #ccc;border-radius:8px}
 button{padding:10px 18px;margin-top:10px;font-size:15px;border:0;border-radius:8px;background:#1f3a5f;color:#fff;cursor:pointer}
 #out{white-space:pre-wrap;background:#f5f5f5;padding:16px;border-radius:8px;margin-top:18px;min-height:44px}
 .row{margin-top:22px;padding-top:16px;border-top:1px solid #eee;color:#666;font-size:13px}
 .row input{font-size:13px}
</style></head><body>
 <h2>RAG Knowledge Assistant</h2>
 <p class="sub">Ask questions about the indexed documents. Answers stream in and are grounded in the sources.</p>
 <input id="q" type="text" placeholder="e.g. What is Reciprocal Rank Fusion?" onkeydown="if(event.key==='Enter')go()">
 <button onclick="go()">Ask</button>
 <div id="out"></div>
 <div class="row">Add a document: <input id="f" type="file" accept=".md,.txt,.pdf">
  <button onclick="up()">Ingest</button> <span id="msg"></span></div>
<script>
async function go(){
  const q=document.getElementById('q').value; if(!q) return;
  const out=document.getElementById('out'); out.textContent='';
  const r=await fetch('/ask/stream',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:q})});
  const reader=r.body.getReader(), dec=new TextDecoder();
  for(;;){const {value,done}=await reader.read(); if(done) break; out.textContent+=dec.decode(value,{stream:true});}
}
async function up(){
  const f=document.getElementById('f').files[0]; if(!f) return;
  const msg=document.getElementById('msg'); msg.textContent='indexing...';
  const fd=new FormData(); fd.append('file',f);
  const r=await fetch('/ingest',{method:'POST',body:fd}); const d=await r.json();
  msg.textContent=d.error?d.error:('indexed — '+d.total_chunks+' chunks in '+d.total_files+' files');
}
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def home():
    return PAGE


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "ingest":
        result = ingest()
        print(f"Index updated: {result['total_chunks']} chunks from {result['total_files']} file(s).")
        if result["changed"]:
            print("  (re)indexed:", ", ".join(result["changed"]))
        if result["removed"]:
            print("  pruned:", ", ".join(result["removed"]))
    elif len(sys.argv) >= 3 and sys.argv[1] == "ask":
        result = ask(" ".join(sys.argv[2:]))
        print("\n" + result["answer"])
        print("\nSources:", ", ".join(result["sources"]) or "—")
    else:
        print(__doc__)
