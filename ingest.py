"""
Ingest pipeline for the FastAPI RAG project.

Right now this just fetches the FastAPI repo so the markdown files
are on disk. Chunking lands in the next commit, embedding after that.
Keeping each step in its own commit so the history reads like a tour
of the pipeline.
"""

import subprocess
import sys
from pathlib import Path

# Pinning the repo URL up here so it's obvious where we're pulling from.
# I'm using the official FastAPI repo on GitHub. If FastAPI ever moves
# this is the one line to change.
REPO_URL = "https://github.com/fastapi/fastapi.git"

# Everything fetched lives under data/. That folder is in .gitignore
# because the docs aren't mine to commit, and re-fetching is cheap.
DATA_DIR = Path(__file__).parent / "data"
REPO_DIR = DATA_DIR / "fastapi"

# The English docs live at this path inside the repo. FastAPI ships
# translations too, but I'm indexing only English for now so my
# retrieval signal isn't diluted by content the embedder wasn't
# trained on.
DOCS_SUBPATH = "docs/en/docs"


def fetch_docs() -> Path:
    """
    Shallow-clone the FastAPI repo into data/fastapi.

    Returns the path to the English docs directory so callers can
    feed it straight into the chunker.

    Shallow (--depth 1) because I don't care about commit history
    for this project, only the current state of the docs. Saves
    most of the download size and skips a bunch of git plumbing.
    """
    DATA_DIR.mkdir(exist_ok=True)

    # Idempotent: if the repo is already here, don't re-clone. This
    # lets me re-run the script during development without waiting
    # on a fresh clone every time.
    if REPO_DIR.exists():
        print(f"Repo already at {REPO_DIR}, skipping clone.")
        return REPO_DIR / DOCS_SUBPATH

    print(f"Cloning {REPO_URL} into {REPO_DIR} ...")
    result = subprocess.run(
        ["git", "clone", "--depth", "1", REPO_URL, str(REPO_DIR)],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        # Surface the real git error rather than a generic message,
        # so when this fails on a network blip or a missing git
        # binary I'm not guessing at the cause.
        print(f"git clone failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)

    docs_path = REPO_DIR / DOCS_SUBPATH
    print(f"Done. English docs are at {docs_path}")
    return docs_path


def main() -> None:
    fetch_docs()


if __name__ == "__main__":
    main()
