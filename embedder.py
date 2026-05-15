"""
Embedder for the FastAPI RAG project.

Reads chunks from data/chunks.jsonl, embeds them with sentence-transformers,
and writes them into a persistent Chroma collection on disk.

Each chunk's prepended-header text becomes the embedding input. The metadata
(source_file, header_path, chunk_index) rides along so retrieval can build
citations later.
"""

from pathlib import Path
import json

from sentence_transformers import SentenceTransformer
import chromadb

# Pinning paths and constants up here so they're easy to find and tweak.
DATA_DIR = Path(__file__).parent / "data"
CHUNKS_PATH = DATA_DIR / "chunks.jsonl"
CHROMA_DIR = Path(__file__).parent / "chroma_db"

# Model and collection names. The collection is the table-equivalent inside
# Chroma. One collection per corpus is plenty for now.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "fastapi_docs"

# Batch size for encode(). Bigger = faster on GPU but more memory. 64 is fine
# on a laptop CPU and never blows up RAM. Tunable later if needed.
BATCH_SIZE = 64


def load_chunks(path: Path) -> list[dict]:
    """Read JSONL into a list of chunk dicts."""
    # This is a bit more code than json.load() but it lets me handle large files without 
    # loading the whole thing into memory at once. If the file is small, it's still fast 
    # enough that the complexity isn't worth worrying about.
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def build_collection(
    chunks: list[dict],
    model: SentenceTransformer,
    client: chromadb.PersistentClient,
) -> int:
    """
    Embed all chunks and write them to a fresh Chroma collection.

    Nukes any existing collection of the same name first (Option A
    idempotency: predictable, simple, slow if you re-run for no reason).

    Returns the number of chunks written so the caller can log it.
    """
    # nuke any existing collection by the same name, then create a new one. 
    try: 
        client.delete_collection(name=COLLECTION_NAME)
    except ValueError:
        pass
    collection = client.create_collection(name=COLLECTION_NAME)

    # pulling parallel lists out of `chunks` so I can batch-encode all documents at once.
    documents = []
    metadatas = []
    ids = []
    for chunk in chunks:
        documents.append(chunk["text"])
        metadatas.append({
            "source_file": chunk["source_file"],
            "header_path": " > ".join(chunk["header_path"]),
            "chunk_index": chunk["chunk_index"],
        })
        ids.append(f"{chunk['source_file']}::{chunk['chunk_index']}")

    # encode all documents in one call. sentence-transformers batches internally, so this is 
    # still memory-efficient and much faster than a Python loop over encode().
    embeddings = model.encode(documents, batch_size=BATCH_SIZE, show_progress_bar=True)
    embeddings = embeddings.tolist()

    """
    IDs are optional but good to have for traceability. I'll use a combination of source file 
    and chunk index to ensure uniqueness and make it easy to track back from an embedding to
    the original text.   
    """
    collection.add(documents=documents, embeddings=embeddings, metadatas=metadatas, ids=ids)

    """
    Note that len(chunks) is the same as len(documents) and len(embeddings) since we're 
    embedding every chunk. If I add any filtering or skipping logic later, I'll want to 
    track the count of embedded chunks separately, but for now this is fine.
    """
    return len(chunks)


def main() -> None:
    print(f"Loading chunks from {CHUNKS_PATH}...")
    chunks = load_chunks(CHUNKS_PATH)
    print(f"Loaded {len(chunks)} chunks.")

    print(f"Loading embedding model {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print(f"Connecting to Chroma at {CHROMA_DIR}...")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    n = build_collection(chunks, model, client)
    print(f"Embedded {n} chunks into collection '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    main()