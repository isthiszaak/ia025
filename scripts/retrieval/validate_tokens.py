"""Audit real-token chunks and full embedding-input budgets without Ollama.

Run from the repository root with the retrieval Python environment. This
validates source slices/token counts; it does not create vectors or quality
measurements. A tokenizer download is needed only on a cold cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from app.retrieval.chunking import chunk_documents  # noqa: E402
from app.retrieval.data import load_documents  # noqa: E402
from app.retrieval.tokenizer import load_tokenizer  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/retrieval")
    parser.add_argument("--tokenizer-cache", type=Path)
    parser.add_argument("--sizes", type=int, nargs="+", default=[256, 512, 1024])
    parser.add_argument("--overlap", type=int, default=50)
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--document-limit", type=int)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if not args.sizes or min(args.sizes) <= 0 or not 0 <= args.overlap < min(args.sizes):
        parser.error("Every chunk size must be positive and overlap smaller than the smallest size")
    if args.max_input_tokens <= 0 or (args.document_limit is not None and args.document_limit <= 0):
        parser.error("Input-token/document limits must be positive")
    prepared = args.data_dir / "prepared"
    manifest = json.loads((prepared / "manifest.json").read_text(encoding="utf-8"))
    corpus_digest = hashlib.sha256()
    with (prepared / "corpus.jsonl").open("rb") as handle:
        while block := handle.read(1024 * 1024):
            corpus_digest.update(block)
    if corpus_digest.hexdigest() != manifest["corpus_sha256"]:
        raise ValueError("Prepared corpus SHA256 differs from the preparation manifest")
    documents = load_documents(prepared)
    total_documents = len(documents)
    if args.document_limit:
        documents = documents[:args.document_limit]
    tokenizer = load_tokenizer(args.tokenizer_cache or args.data_dir / "tokenizer")
    report = {
        "verification": "real tokenizer, exact source slices; no embeddings or Ollama requests",
        "dataset": manifest["dataset"], "revision": manifest["revision"],
        "corpus_sha256": manifest["corpus_sha256"],
        "document_count_total": total_documents, "document_count_validated": len(documents),
        "tokenizer": tokenizer.identity(), "embedding_profile": "embeddinggemma",
        "embedding_max_input_tokens": args.max_input_tokens,
        "configurations": [], "failures": [],
    }
    started = time.perf_counter()
    for size in args.sizes:
        row = {
            "chunk_size": size, "chunk_unit": "tokens", "chunk_overlap": args.overlap,
            "chunk_count": 0, "max_content_tokens": 0,
            "max_embedding_input_tokens": 0, "adjusted_overlap_chunks": 0,
            "overlap_adjustments": [],
        }
        for document_index, document in enumerate(documents):
            covered = 0
            try:
                for chunk in chunk_documents([document], size, args.overlap,
                                             chunk_unit="tokens", tokenizer=tokenizer):
                    metadata = chunk.metadata
                    start, end = metadata["start_char"], metadata["end_char"]
                    if chunk.text != document.text[start:end] or not start <= covered < end:
                        raise ValueError(f"Source coverage/offset mismatch at {start}:{end}")
                    if chunk.document_id != document.id or chunk.source != document.source:
                        raise ValueError("Source identity mismatch")
                    content_count = tokenizer.count(chunk.text, add_special_tokens=False)
                    if not content_count == metadata["token_count"] <= size:
                        raise ValueError(f"Content token budget exceeded: {content_count} > {size}")
                    full_input = f"title: {chunk.title or 'none'} | text: {chunk.text}"
                    input_count = tokenizer.count(full_input, add_special_tokens=True)
                    if input_count > args.max_input_tokens:
                        raise ValueError(f"Complete embedding input exceeds limit: {input_count}")
                    if metadata["chunk_index"] and metadata["actual_overlap_tokens"] != args.overlap:
                        row["adjusted_overlap_chunks"] += 1
                        row["overlap_adjustments"].append({
                            "document_id": document.id, "chunk_id": chunk.id,
                            "start_char": start, "end_char": end,
                            "actual_overlap_tokens": metadata["actual_overlap_tokens"],
                        })
                    row["chunk_count"] += 1
                    row["max_content_tokens"] = max(row["max_content_tokens"], content_count)
                    row["max_embedding_input_tokens"] = max(row["max_embedding_input_tokens"], input_count)
                    covered = end
                if covered != len(document.text):
                    raise ValueError(f"Missing document tail: {covered} != {len(document.text)}")
            except Exception as exc:
                report["failures"].append({"chunk_size": size, "document_id": document.id, "error": str(exc)})
            if (document_index + 1) % 500 == 0:
                print(f"size={size}: {document_index + 1}/{len(documents)} documents checked; "
                      f"failures={len(report['failures'])}", flush=True)
        report["configurations"].append(row)
        print(json.dumps({key: value for key, value in row.items() if key != "overlap_adjustments"}), flush=True)
    report["elapsed_seconds"] = time.perf_counter() - started
    report["status"] = "ok" if not report["failures"] else "error"
    destination = args.report or args.data_dir / "token_validation.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{report['status']}: report saved to {destination}", flush=True)
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
