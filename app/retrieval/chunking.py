"""Character and real-token chunking with stable IDs and exact source slices."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
import hashlib
import json
import re
from typing import Any, Iterable, Iterator

from .data import Document


CHUNKING_VERSION = "characters-boundaries-v1"
TOKEN_CHUNKING_VERSION = "tokens-offsets-v1"


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
    *,
    chunk_unit: str = "characters",
    tokenizer=None,
) -> Iterator[Chunk]:
    """Yield bounded exact text slices with source metadata.

    Offsets use Python Unicode characters, not UTF-8 bytes or model tokens.
    In legacy character mode, natural boundaries are preferred. Token mode
    requires an explicit tokenizer and uses its offsets and real token counts.
    Both modes retain exact slices of the normalized source document.
    """
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if isinstance(chunk_overlap, bool) or not isinstance(chunk_overlap, int) or not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must be an integer with 0 <= overlap < chunk_size")
    if chunk_unit not in {"characters", "tokens"}:
        raise ValueError("chunk_unit must be characters or tokens")
    if chunk_unit == "tokens":
        if tokenizer is None:
            raise ValueError("Token chunking requires an explicit tokenizer")
        yield from _token_chunks(documents, chunk_size, chunk_overlap, tokenizer)
        return
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


def _token_chunks(documents, chunk_size, chunk_overlap, tokenizer):
    """Slice at tokenizer offsets, retaining whitespace and every source character.

    Token windows refer to the encoding of the complete normalized document.
    No decoded/re-encoded substitute text is used. A boundary inside a Unicode
    byte-fallback group is moved forward for overlap, or backward for chunk end;
    ``actual_overlap_tokens`` makes that representational adjustment explicit.
    Each exact slice is counted again so boundary effects cannot exceed the
    requested token budget. Chunk sizes exclude model prefixes/special tokens.
    """
    identity = tokenizer.identity()
    for document in documents:
        if not isinstance(document.text, str) or not document.text:
            continue
        encoding = tokenizer.encode(document.text, add_special_tokens=False)
        offsets = encoding.offsets
        count = len(encoding.ids)
        # A tokenizer may omit whitespace or normalization characters. Attaching
        # each gap to the preceding span preserves the exact original document.
        boundaries = {0: 0, count: len(document.text)}
        last_boundary_char = 0
        for index in range(1, count):
            previous_end = offsets[index - 1][1]
            current_start = offsets[index][0]
            if previous_end <= current_start and last_boundary_char < current_start < len(document.text):
                boundaries[index] = current_start
                last_boundary_char = current_start
        starts = sorted(boundaries)
        start_token = 0
        previous_end_token = 0
        chunk_index = 0
        while start_token < count or (count == 0 and chunk_index == 0):
            start_char = boundaries[start_token] if count else 0
            candidates = starts[bisect_right(starts, start_token):bisect_right(starts, start_token + chunk_size)]
            if count == 0:
                candidates = [0]
            for end_token in reversed(candidates):
                end_char = boundaries[end_token]
                text = document.text[start_char:end_char]
                token_count = tokenizer.count(text, add_special_tokens=False)
                if token_count <= chunk_size:
                    break
            else:
                raise ValueError(
                    f"chunk_size={chunk_size} cannot fit a complete Unicode/token span "
                    f"at character {start_char} in document {document.id}"
                )
            metadata = {
                **document.metadata, "start_char": start_char, "end_char": end_char,
                "start_token": start_token, "end_token": end_token,
                "token_count": token_count, "chunk_index": chunk_index,
                "chunk_unit": "tokens", "chunking_version": TOKEN_CHUNKING_VERSION,
                "tokenizer": identity, "chunk_overlap": chunk_overlap,
                "actual_overlap_tokens": previous_end_token - start_token if chunk_index else 0,
            }
            chunk_identity = {
                "version": TOKEN_CHUNKING_VERSION, "document_id": document.id,
                "chunk_unit": "tokens", "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap, "tokenizer": identity,
                "start_char": start_char, "end_char": end_char,
                "start_token": start_token, "end_token": end_token, "text": text,
            }
            chunk_id = "chunk_" + hashlib.sha256(
                json.dumps(chunk_identity, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":")).encode("utf-8")
            ).hexdigest()
            yield Chunk(chunk_id, document.id, text, document.source, document.title, metadata)
            if end_token == count:
                break
            desired = end_token - chunk_overlap
            next_index = bisect_left(starts, max(start_token + 1, desired))
            if next_index >= len(starts) or starts[next_index] > end_token:
                raise ValueError(f"Token chunking made no progress for document {document.id}")
            previous_end_token = end_token
            start_token = starts[next_index]
            chunk_index += 1
