"""
Agentic RAG — a LangGraph agent that decides when to search the knowledge base.

Unlike plain RAG (retrieve once, then answer), this is an *agent*: it reasons about
the question, calls tools when it needs to, reads the results, and can call again
before answering. Built with LangGraph's ReAct agent + the same local FAISS index
and local LLM (Ollama) as rag.py.

Tools the agent can call:
  - search_documents(query): hybrid + re-ranked search over the FAISS knowledge base
  - list_sources(): list the filenames currently indexed

Run:
  python agent.py "List your documents, then explain how a cross-encoder re-ranker works."
"""
import sys

from langchain.agents import create_agent
from langchain_core.tools import tool

from rag import load_index, retrieve, _cite, get_llm


@tool
def search_documents(query: str) -> str:
    """Search the knowledge base and return the most relevant passages for a query."""
    items = retrieve(query)  # full hybrid + re-rank pipeline from rag.py
    if not items:
        return "No index found. Run:  python rag.py ingest"
    return "\n\n".join(f"[{_cite(d)}] {d.page_content}" for d, _ in items)


@tool
def list_sources() -> str:
    """List the filenames of all documents currently in the knowledge base."""
    store = load_index()
    if store is None:
        return "No index found. Run:  python rag.py ingest"
    sources = sorted({d.metadata.get("source", "?") for d in store.docstore._dict.values()})
    return ", ".join(sources) or "none"


SYSTEM_PROMPT = (
    "You are a research assistant with access to a document knowledge base. "
    "Use the tools to look things up before answering — do not rely on prior knowledge. "
    "Cite the source you used in square brackets, e.g. [retrieval.md › Cross-encoder re-ranking]. "
    "If the answer is not in the documents, say so plainly."
)


def build_agent():
    """A ReAct-style agent: LLM + tools, wired as a LangGraph state machine."""
    return create_agent(get_llm(), tools=[search_documents, list_sources], system_prompt=SYSTEM_PROMPT)


def run(question: str) -> str:
    agent = build_agent()
    result = agent.invoke({"messages": [("user", question)]})
    return result["messages"][-1].content


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or (
        "List your documents, then explain how a cross-encoder re-ranker works."
    )
    print("Q:", question, "\n")
    print(run(question))
