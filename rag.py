"""
RAG Knowledge Assistant — a small, real Retrieval-Augmented Generation app.

What it does:
  1. Loads .md / .txt / .pdf files from ./data
  2. Splits them into chunks and embeds them with a local, free model
  3. Stores the vectors in a FAISS index (saved to ./.index)
  4. On a question: retrieves the most relevant chunks and asks an LLM to
     answer using ONLY those chunks, citing the source filenames.

Run:
  python rag.py ingest              # build the index from ./data
  python rag.py ask "your question" # ask from the command line
  uvicorn rag:app --reload          # web API + tiny chat UI -> http://localhost:8000
"""
from __future__ import annotations  # lets `X | None` hints work on Python 3.9

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
INDEX_DIR = os.getenv("INDEX_DIR", ".index")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))
TOP_K = int(os.getenv("TOP_K", "4"))
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")  # "ollama" = fully local & open-source

# The system prompt is what keeps answers grounded: only use retrieved context,
# admit when the answer isn't there (this is how RAG reduces hallucination).
SYSTEM_PROMPT = (
    "You answer questions using ONLY the provided context.\n"
    'If the answer is not in the context, say: "I don\'t have that in the provided documents."\n'
    "Be concise. Cite the source filename(s) you used in square brackets, e.g. [file.md]."
)


def get_embeddings():
    """Local, free embedding model. Downloads once (~80MB) on first use."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def get_llm():
    """The generation model. Local open-source (Ollama) by default; cloud optional."""
    if LLM_PROVIDER == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-6"), temperature=0)
    if LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)
    # Default: a local open-source model served by Ollama. No API key, runs offline.
    from langchain_ollama import ChatOllama
    return ChatOllama(model=os.getenv("OLLAMA_MODEL", "llama3.2"), temperature=0)


def load_docs() -> list[Document]:
    """Read every supported file in DATA_DIR into a LangChain Document."""
    docs: list[Document] = []
    for path in sorted(DATA_DIR.rglob("*")):
        suffix = path.suffix.lower()
        if suffix not in {".md", ".txt", ".pdf"}:
            continue
        if suffix == ".pdf":
            from pypdf import PdfReader
            text = "\n".join((page.extract_text() or "") for page in PdfReader(str(path)).pages)
        else:
            text = path.read_text(encoding="utf-8", errors="ignore")
        if text.strip():
            docs.append(Document(page_content=text, metadata={"source": path.name}))
    return docs


def ingest() -> int:
    """Load -> split -> embed -> save FAISS index."""
    docs = load_docs()
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)
    if not chunks:
        print(f"No documents found in {DATA_DIR}/. Add some .md/.txt/.pdf files.")
        return 0
    store = FAISS.from_documents(chunks, get_embeddings())
    store.save_local(INDEX_DIR)
    print(f"Indexed {len(chunks)} chunks from {len(docs)} document(s) into '{INDEX_DIR}/'.")
    return len(chunks)


def load_index() -> FAISS | None:
    if not Path(INDEX_DIR).exists():
        return None
    # allow_dangerous_deserialization is safe here: we built this index ourselves.
    return FAISS.load_local(INDEX_DIR, get_embeddings(), allow_dangerous_deserialization=True)


def ask(question: str) -> dict:
    """Retrieve top-k chunks, then have the LLM answer from them with citations."""
    store = load_index()
    if store is None:
        return {"answer": "No index found. Run:  python rag.py ingest", "sources": []}
    hits = store.similarity_search(question, k=TOP_K)
    context = "\n\n---\n\n".join(
        f"[source: {d.metadata['source']}]\n{d.page_content}" for d in hits
    )
    reply = get_llm().invoke([
        ("system", SYSTEM_PROMPT),
        ("human", f"Context:\n{context}\n\nQuestion: {question}"),
    ])
    sources = sorted({d.metadata["source"] for d in hits})
    return {"answer": reply.content, "sources": sources}


# --------------------------------------------------------------------------
# Web API (optional) — run with:  uvicorn rag:app --reload
# --------------------------------------------------------------------------
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

app = FastAPI(title="RAG Knowledge Assistant")


class Question(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok", "indexed": Path(INDEX_DIR).exists(), "provider": LLM_PROVIDER}


@app.post("/ask")
def ask_endpoint(q: Question):
    return ask(q.question)


PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>RAG Assistant</title>
<style>
 body{font-family:system-ui,sans-serif;max-width:680px;margin:48px auto;padding:0 16px;color:#1a1a1a}
 h2{margin-bottom:4px} p.sub{color:#666;margin-top:0}
 input{width:100%;padding:11px;font-size:15px;border:1px solid #ccc;border-radius:8px}
 button{padding:10px 18px;margin-top:10px;font-size:15px;border:0;border-radius:8px;background:#1f3a5f;color:#fff;cursor:pointer}
 #out{white-space:pre-wrap;background:#f5f5f5;padding:16px;border-radius:8px;margin-top:18px;min-height:44px}
 .src{color:#666;font-size:13px;margin-top:10px}
</style></head><body>
 <h2>RAG Knowledge Assistant</h2>
 <p class="sub">Ask questions about the indexed documents. Answers are grounded in the sources.</p>
 <input id="q" placeholder="e.g. How does YourBackers protect video content?" onkeydown="if(event.key==='Enter')go()">
 <button onclick="go()">Ask</button>
 <div id="out"></div>
<script>
async function go(){
  const q=document.getElementById('q').value; if(!q) return;
  const out=document.getElementById('out'); out.textContent='Thinking...';
  const r=await fetch('/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:q})});
  const d=await r.json();
  const safe=d.answer.replace(/&/g,'&amp;').replace(/</g,'&lt;');
  out.innerHTML=safe+'<div class="src">Sources: '+(d.sources.join(', ')||'—')+'</div>';
}
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def home():
    return PAGE


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "ingest":
        ingest()
    elif len(sys.argv) >= 3 and sys.argv[1] == "ask":
        result = ask(" ".join(sys.argv[2:]))
        print("\n" + result["answer"])
        print("\nSources:", ", ".join(result["sources"]) or "—")
    else:
        print(__doc__)
