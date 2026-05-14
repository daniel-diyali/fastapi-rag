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

An Anthropic API key will be needed once I hit Phase 4 (generation). Instructions for the `.env` file go here when I get there.

## Learning journal

I'll fill this in as I work through each phase. Each entry should answer: what was the decision, what alternatives did I consider, what did I pick and why, and (later) what did the evals say about that pick.

## Chunking

I'm chunking the FastAPI English docs from `docs/en/docs` in [github.com/fastapi/fastapi](https://github.com/fastapi/fastapi). About 700 markdown files raw, 835 chunks once I'm done. I excluded the `reference/` folder (mkdocstrings-generated stubs that don't say anything in raw form), `release-notes.md`, `external-links.md`, `newsletter.md`, `fastapi-people.md`, and any file starting with `_`. None of those are "how do I use FastAPI" content, and including them would dilute the retrieval signal.

### How

Header-aware split on `## `. Each chunk gets the document title and section heading prepended to its text, so a chunk from the "Wildcards" section of the CORS doc starts with "CORS > Wildcards" before the body. If a section is over 400 words, I sub-split on paragraph boundaries with a 300-word target. Code fences are atomic. They never get split mid-block, because cutting a Python example in half makes the chunk useless for retrieval and useless for the LLM that has to read it later.

Why 300/400? The embedder I'm starting with (`all-MiniLM-L6-v2`) silently truncates anything past 256 tokens. 300 English words is roughly 200-220 tokens, comfortably under the cap. If a chunk slips through at 400 words, that's still close enough that I'd rather emit it whole than slice prose mid-sentence.

### Alternatives I considered

**Fixed-size token windows** (a 512-token sliding window, for example). Easiest to write. I rejected it because it ignores document structure entirely. A chunk could start mid-sentence and end mid-code-block, which is exactly the failure mode header-aware splitting is built to avoid.

**Pure paragraph splitting.** Better than fixed-size, but a paragraph by itself doesn't tell the embedder what topic it's about. Headers are free topic anchors. Throwing them away to gain simplicity felt wrong.

**Semantic chunking** (split where embedding similarity drops between sentences). Cooler in theory and probably better in practice, but it adds an extra embedder pass to ingest and a layer of tuning I don't need yet. Punted unless evals show header-aware can't carry the load.

### Why I prepended the header path into the chunk text

This is the one decision I want to flag because I think it's the biggest lever on retrieval quality before evals are even running.

A chunk's metadata stores the header path separately, but the embedder doesn't see metadata. It only sees the text. If the body of a section says "Use `allow_origins=['*']` to allow any origin" and never says the words "CORS" or "wildcards," the resulting vector lands in some neutral zone, and a query like "how do I configure CORS wildcards?" can't find it.

By prepending "CORS > Wildcards" into the chunk's text, I'm baking the topic words into the embedding. The vector lands closer to where CORS-related question vectors live. I'll ablate this in Phase 6 (chunk once with the prefix, once without, compare hit@3) and document the actual numbers.

### What I'm watching for

A few rough edges I noticed while eyeballing the JSONL output. Leaving as TODOs until evals tell me whether they matter:

- The H1 line stays in the body of the intro chunk, which means the doc title appears twice in that chunk's text (once in the prefix, once in the H1). Slight redundancy, probably fine, but worth measuring.
- A small number of sections over 400 words consist of one giant paragraph with no break. Those end up as a single oversized chunk. Sentence-level fallback would fix it but felt premature.
- Some advanced docs are mostly code fences with thin prose. Those chunks may produce embeddings biased toward syntax instead of semantics.

### Why JSONL between phases

The chunker writes every chunk as one JSON object per line in `data/chunks.jsonl`. The embedder will read that file in Phase 2. I'm putting the file on disk between phases on purpose. That way I can iterate on the embedder without re-running the chunker, and I can `head` the file and read chunks with my eyes before trusting them. The H1 redundancy above is exactly the kind of thing you only catch by looking.

## Eval results

I'll only know if this system actually works once I have an evaluation harness measuring it. Phase 6 builds that harness. Results land in this table as I iterate.

| Iteration | What changed | hit@1 | hit@3 | MRR | Faithfulness | Helpfulness |
|-----------|--------------|-------|-------|-----|--------------|-------------|
| _(none yet)_ |  |  |  |  |  |  |


