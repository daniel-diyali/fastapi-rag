# TODO: H1 line stays in the intro chunk body, doubling up the doc title
# with the prepended prefix. Probably fine but might inflate weight in
# the embedding. Revisit if evals show it.

"""
Markdown chunker for the FastAPI RAG project.

Splits each markdown file on H2 (##) boundaries. If a section is bigger
than the word budget, sub-splits it on paragraph breaks. Never splits
inside a fenced code block, because cutting Python in half makes the
chunk useless.

Each chunk carries metadata (source_file, header_path, chunk_index) so
the retriever can build citations later, and the document title and
section heading are prepended to the chunk text so the embedder knows
what topic the chunk is about even when the body doesn't say it directly.
"""

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator
import re


# I'm targeting ~300 words because the embedder I'm starting with
# (all-MiniLM-L6-v2) silently truncates anything past 256 tokens.
# 300 English words is roughly 200-220 tokens, safely under the cap.
# Hard cap at 400 words so a long single section doesn't sneak through.
TARGET_WORDS = 300
HARD_CAP_WORDS = 400

# Deny list applied during corpus walk.
#   reference/  -> mkdocstrings :::-directives, useless in raw form
#   _*          -> private/test files like _llm-test.md
#   release-notes.md -> changelog noise, would pollute retrieval with version numbers
#   meta files  -> not "how do I use FastAPI" content
EXCLUDE_DIRS = {"reference"}
EXCLUDE_FILES = {
    "release-notes.md",
    "external-links.md",
    "newsletter.md",
    "fastapi-people.md",
}
EXCLUDE_PREFIX = "_"


@dataclass
class Chunk:
    """One piece of documentation, ready to be embedded.

    `text` is what we'll feed to the embedder. The doc title and header
    path are prepended so the resulting vector knows what topic this
    chunk is about, even if the body never says the word.
    """
    text: str
    source_file: str         # path relative to the docs root, e.g. tutorial/cors.md
    header_path: list[str]   # e.g. ["CORS", "Wildcards"]
    chunk_index: int         # 0-based ordinal within the source file

    def to_dict(self) -> dict:
        return asdict(self)


# FastAPI's docs add anchor suffixes to headers, like
#   ## Wildcards { #wildcards }
# That trailing { #... } is useful for building citation links later but
# it's pure noise in the embedded text, so I strip it before chunking.
_ANCHOR_RE = re.compile(r"\s*\{\s*#[^}]+\}\s*$")


def _strip_anchor(header: str) -> str:
    return _ANCHOR_RE.sub("", header).strip()


def _word_count(text: str) -> int:
    # Approximating tokens with word count. For all-MiniLM-L6-v2 the
    # ratio is roughly 1 word -> ~1.3 tokens for English. Not exact,
    # but precise enough for "is this chunk going to get truncated?"
    return len(text.split())


def parse_sections(md_text: str) -> list[tuple[str, str]]:
    """Split a markdown file on H2 boundaries.

    Returns a list of (h2_title, body) pairs. The intro before the first
    H2, if any, gets an empty title (the H1 will be prepended later).

    I'm walking line by line with a state machine instead of running a
    regex over the whole file because headers inside fenced code blocks
    must NOT trigger a split. Python comments and shell prompts use # too,
    and a regex like /^## / would happily slice them. The cost of being
    stateful is ten extra lines. The cost of being naive is broken chunks.
    """
    sections: list[tuple[str, str]] = []
    current_title = ""
    current_body: list[str] = []
    in_code_block = False

    for line in md_text.splitlines():
        # Toggle on any fence line: ``` or ```python both count.
        if line.lstrip().startswith("```"):
            in_code_block = not in_code_block
            current_body.append(line)
            continue

        # Only treat ## as a section break when we're outside code.
        if not in_code_block and line.startswith("## "):
            if current_title or current_body:
                sections.append((current_title, "\n".join(current_body).strip()))
            current_title = _strip_anchor(line[3:])
            current_body = []
            continue

        current_body.append(line)

    # Don't forget the trailing section.
    if current_title or current_body:
        sections.append((current_title, "\n".join(current_body).strip()))

    return sections


def _extract_h1(md_text: str) -> str:
    """Pull the first H1 (#) line as the document title.

    Falls back to "" if there's no H1. That shouldn't happen in FastAPI
    docs, but I'd rather degrade gracefully than crash on the one file
    that breaks the pattern.
    """
    for line in md_text.splitlines():
        if line.startswith("# "):
            return _strip_anchor(line[2:])
    return ""


def _split_oversized(body: str, max_words: int) -> list[str]:
    """Break a too-long section into smaller pieces on paragraph breaks.

    Greedy bin-packing: keep adding paragraphs to the current piece until
    one more would push it over max_words, then start a new piece. Treats
    a fenced code block as a single atomic "paragraph" so it never gets
    split in the middle.

    Edge case: a single paragraph (or code block) that's already larger
    than max_words ends up as its own oversized piece. I'd rather emit
    one coherent chunk over the budget than slice prose mid-sentence.
    If evals show this matters, I'll add a sentence-level fallback.
    """
    # First pass: group lines into paragraphs, treating code fences as atomic.
    paragraphs: list[str] = []
    in_code = False
    buf: list[str] = []

    for line in body.splitlines():
        if line.lstrip().startswith("```"):
            in_code = not in_code
            buf.append(line)
            continue
        if not in_code and line.strip() == "":
            if buf:
                paragraphs.append("\n".join(buf).strip())
                buf = []
        else:
            buf.append(line)
    if buf:
        paragraphs.append("\n".join(buf).strip())

    # Second pass: greedy pack paragraphs into pieces under the cap.
    pieces: list[str] = []
    current: list[str] = []
    current_words = 0
    for para in paragraphs:
        para_words = _word_count(para)
        if current and current_words + para_words > max_words:
            pieces.append("\n\n".join(current))
            current = [para]
            current_words = para_words
        else:
            current.append(para)
            current_words += para_words
    if current:
        pieces.append("\n\n".join(current))

    return pieces


def chunk_markdown_file(md_path: Path, source_relpath: str) -> list[Chunk]:
    """Turn one markdown file into a list of Chunks."""
    md_text = md_path.read_text(encoding="utf-8")
    doc_title = _extract_h1(md_text)
    sections = parse_sections(md_text)

    chunks: list[Chunk] = []
    chunk_idx = 0

    for h2_title, body in sections:
        if not body.strip():
            continue

        # Header path skips empty parts. The intro section (before the
        # first H2) has an empty h2_title, so its path is just [doc_title].
        header_path = [t for t in (doc_title, h2_title) if t]

        # Under budget -> one chunk. Over -> split on paragraphs.
        bodies = (
            [body]
            if _word_count(body) <= HARD_CAP_WORDS
            else _split_oversized(body, TARGET_WORDS)
        )

        for piece in bodies:
            # Prepend the header path to the chunk's TEXT, not just its
            # metadata. The embedder needs to see the topic words to
            # produce a vector that points the right direction in space,
            # otherwise a section like "Wildcards" reads as topic-less.
            prefix = " > ".join(header_path)
            text = f"{prefix}\n\n{piece}" if prefix else piece

            chunks.append(Chunk(
                text=text,
                source_file=source_relpath,
                header_path=header_path,
                chunk_index=chunk_idx,
            ))
            chunk_idx += 1

    return chunks


def _should_include(path: Path, docs_root: Path) -> bool:
    """Apply the deny list."""
    rel = path.relative_to(docs_root)
    if any(part in EXCLUDE_DIRS for part in rel.parts):
        return False
    if rel.name in EXCLUDE_FILES or rel.name.startswith(EXCLUDE_PREFIX):
        return False
    return True


def chunk_corpus(docs_root: Path) -> Iterator[Chunk]:
    """Walk every included markdown file under docs_root and yield chunks.

    I'm yielding instead of returning a list so the ingest pipeline can
    stream chunks straight into the JSONL writer (or, later, the embedder)
    without holding the whole corpus in memory. Also sorting paths so
    runs are deterministic and `git diff` on the chunks output is sane.
    """
    for md_path in sorted(docs_root.rglob("*.md")):
        if not _should_include(md_path, docs_root):
            continue
        relpath = str(md_path.relative_to(docs_root))
        yield from chunk_markdown_file(md_path, relpath)
