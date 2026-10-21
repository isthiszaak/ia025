"""Character-based legal-text chunking with stable IDs and exact offsets."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import re
from typing import Any, Iterable, Iterator

from .data import Document


CHUNKING_VERSION = "characters-boundaries-v1"


@dataclass(frozen=True)
class Chunk:
    id: str
    document_id: str
    text: str
    source: str
    title: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _end_boundary(text: str, start: int, limit: int, overlap: int) -> int:
    if limit == len(text):
        return limit
    # Keep a useful amount of new content and guarantee progress even for large
    # overlaps. All separators remain inside an exact document slice.
    minimum = start + max(overlap + 1, (limit - start) // 2)
    window = text[start:limit]
    for pattern in (r"\n\s*\n", r"[.!?](?:[ \t]+|\n)", r"\n", r"[ \t]+"):
        positions = [start + match.end() for match in re.finditer(pattern, window)]
        usable = [position for position in positions if position >= minimum]
        if usable:
            return usable[-1]
    return limit


def chunk_documents(
    documents: Iterable[Document],
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
) -> Iterator[Chunk]:
    """Yield bounded character slices with exact overlap and source metadata.

    Offsets use Python Unicode characters, not UTF-8 bytes or model tokens.
    ``chunk_overlap`` characters from each previous chunk are repeated. Natural
    boundaries are preferred in the latter half; long units are hard-split.
    """
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer number of characters")
    if isinstance(chunk_overlap, bool) or not isinstance(chunk_overlap, int) or not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must be an integer with 0 <= overlap < chunk_size")
    for document in documents:
        if not isinstance(document.text, str) or not document.text:
            continue
        start = 0
        chunk_index = 0
        while start < len(document.text):
            end = _end_boundary(document.text, start, min(start + chunk_size, len(document.text)), chunk_overlap)
            text = document.text[start:end]
            identity = {
                "version": CHUNKING_VERSION, "document_id": document.id,
                "chunk_size": chunk_size, "chunk_overlap": chunk_overlap,
                "start_char": start, "end_char": end, "text": text,
            }
            chunk_id = "chunk_" + hashlib.sha256(
                json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            yield Chunk(
                id=chunk_id, document_id=document.id, text=text,
                source=document.source, title=document.title,
                metadata={
                    **document.metadata, "start_char": start, "end_char": end,
                    "chunk_index": chunk_index, "chunking_version": CHUNKING_VERSION,
                },
            )
            if end == len(document.text):
                break
            start = end - chunk_overlap
            chunk_index += 1
