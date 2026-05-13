# fastapi-rag

A retrieval-augmented question-answering system over the FastAPI documentation. You ask "how do I add CORS middleware?", it finds the relevant chunks of the docs, hands them to Claude, and returns an answer with citations back to the source files.

## Why I'm building this

I'm a CS undergrad and I've never built a RAG pipeline before. I want to understand every layer of the system (chunking, embeddings, retrieval, prompt design, evaluation) well enough to make informed trade-offs at each step, instead of calling a high-level abstraction and hoping for the best. So I'm building this without LangChain.

This README doubles as a learning journal. Every meaningful decision gets a note here about what I tried, what didn't work, and why I settled on what I did. The interesting story isn't "I built a RAG system." It's "I tried strategy X, my eval said hit@3 was 0.62, I tried strategy Y, it went to 0.81, here's why that happened."

## Stack

- Python 3.11+
- sentence-transformers for embeddings (local, free). Behind an interface so I can swap to OpenAI or Voyage later without rewriting the retriever.
- ChromaDB as the vector store. Picked over FAISS because Chroma gives me metadata filtering and persistence out of the box, and I need both for citations.
- Anthropic Claude for generation.
- FastAPI for the query endpoint.
- pytest for the eval harness.
- Docker + AWS Lambda for deployment (stretch goal).
- GitHub Actions for CI that fails the build if eval scores regress (stretch goal).

## Status

Phase 0 done. Repo skeleton, license, gitignore, README.

## Setup

```bash
git clone https://github.com/<your-username>/fastapi-rag.git
cd fastapi-rag
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

An Anthropic API key will be needed once we hit Phase 4 (generation). Instructions for the `.env` file go here when we get there.

## Learning journal

I'll fill this in as I work through each phase. Each entry should answer: what was the decision, what alternatives did I consider, what did I pick and why, and (later) what did the evals say about that pick.

_Empty until Phase 1._

## Eval results

I'll only know if this system actually works once I have an evaluation harness measuring it. Phase 6 builds that harness. Results land in this table as I iterate.

| Iteration | What changed | hit@1 | hit@3 | MRR | Faithfulness | Helpfulness |
|-----------|--------------|-------|-------|-----|--------------|-------------|
| _(none yet)_ |  |  |  |  |  |  |
