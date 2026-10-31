"""Reproducible one-factor-at-a-time runs of retrieval, without generation."""

from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from .index import Retriever, build_index, dependency_versions


CHUNK_SIZES = (256, 512, 1024)
TOP_K_VALUES = (1, 3, 5)
CASE_FIELDS = ("case_id", "user_input", "reference", "references",
               "all_formatted_references", "all_formated_references")


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
                    encoding="utf-8")


def corpus_snapshot(config):
    """Record requested corpus identity even when no index can be opened."""
    prepared = config.data_dir / "prepared"
    snapshot = {"prepared_dir": str(prepared), "selection": "all" if config.corpus_selection is None
                else str(config.corpus_selection)}
    for name, path in (("corpus", prepared / "corpus.jsonl"),
                       ("selection", config.corpus_selection)):
        if path is not None:
            try:
                snapshot[name + "_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            except OSError as exc:
                snapshot[name + "_error"] = f"{type(exc).__name__}: {exc}"
    try:
        snapshot["prepared_manifest"] = json.loads(
            (prepared / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        snapshot["manifest_error"] = f"{type(exc).__name__}: {exc}"
    return snapshot


def _configurations(config, fixed_top_k, fixed_chunk_size):
    if config.chunk_unit != "tokens":
        raise ValueError("As comparações exigem chunk_unit=tokens")
    if not 0 <= config.chunk_overlap < min(CHUNK_SIZES):
        raise ValueError("O overlap constante deve ser menor que 256 tokens")
    if isinstance(fixed_top_k, bool) or fixed_top_k not in TOP_K_VALUES:
        raise ValueError("fixed_top_k deve ser 1, 3 ou 5")
    if isinstance(fixed_chunk_size, bool) or fixed_chunk_size not in CHUNK_SIZES:
        raise ValueError("fixed_chunk_size deve ser 256, 512 ou 1024 tokens")
    combinations = [(size, fixed_top_k) for size in CHUNK_SIZES]
    combinations += [(fixed_chunk_size, top_k) for top_k in TOP_K_VALUES]
    return [replace(config, chunk_size=size, top_k=top_k)
            for size, top_k in dict.fromkeys(combinations)]


def run_comparisons(config, cases, output_dir, fixed_top_k=3, fixed_chunk_size=512, build=False):
    """Run five planned settings and persist successes and failures for each case.

    ``cases`` is the same validated list of case_id/user_input dictionaries used
    by the query CLI. Only user_input reaches the retriever. All configurations
    retain the caller's overlap, corpus, tokenizer and embedding settings. There
    are no relevance or answer-quality metrics. Index construction is opt-in and
    uses the already prepared corpus; changes to top-k reuse the open index.

    Returns a JSON-compatible summary, including the unique output directory and
    total error count. Artifacts are written even if index build/open fails.
    """
    configurations = _configurations(config, fixed_top_k, fixed_chunk_size)
    cases = list(cases)
    if not cases or any(not isinstance(case, dict)
                        or not isinstance(case.get("user_input"), str)
                        or not case["user_input"].strip() for case in cases):
        raise ValueError("Informe casos com user_input não vazio")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid4().hex[:8]
    destination = Path(output_dir) / run_id
    destination.mkdir(parents=True, exist_ok=False)
    common = {"run_id": run_id, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "stage": "retrieval_only", "protocol": "one_factor_at_a_time",
              "fixed_top_k": fixed_top_k, "fixed_chunk_size": fixed_chunk_size,
              "chunk_overlap_tokens": config.chunk_overlap,
              "build_indexes": bool(build), "dependencies": dependency_versions(),
              "corpus": corpus_snapshot(config), "cases_count": len(cases),
              "cases_sha256": hashlib.sha256(json.dumps(cases, ensure_ascii=False,
                  sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()}
    summary = {**common, "output_dir": str(destination), "comparisons_count": len(configurations),
               "retrieval_count": 0, "errors": 0, "state": "running", "runs": []}
    _write_json(destination / "manifest.json", summary)
    # One open collection (and, optionally, one build) per chunk size, including
    # a cached failure so changing top-k cannot silently trigger another build.
    indexes = {}
    for current in configurations:
        setting = f"tokens-{current.chunk_size}-topk-{current.top_k}"
        directory = destination / setting
        directory.mkdir()
        if current.chunk_size not in indexes:
            started = perf_counter()
            state = {"retriever": None, "index": None, "error": None, "error_stage": None}
            try:
                if build:
                    state["error_stage"] = "build"
                    state["index"] = build_index(current)
                state["error_stage"] = "open"
                state["retriever"] = Retriever(current)
                state["index"] = state["retriever"].manifest
                state["error_stage"] = None
            except Exception as exc:
                state["error"] = exc
            state["setup_seconds"] = perf_counter() - started
            indexes[current.chunk_size] = state
        state = indexes[current.chunk_size]
        retriever = state["retriever"]
        index = state["index"]
        collection = getattr(retriever, "collection_name", None)
        model = {"name": current.model, "profile": current.profile,
                 "dimension": current.dimension,
                 "digest": index.get("model_digest") if index else None}
        configuration = asdict(current)
        rows = []
        for case in cases:
            row = {key: case[key] for key in CASE_FIELDS if key in case}
            row.update(config=configuration, collection=collection, model=model,
                       chunk_size=current.chunk_size, chunk_unit=current.chunk_unit,
                       top_k=current.top_k, retrieved_chunks=[], retrieved_contexts=[],
                       latency_seconds=None)
            if state["error"] is not None:
                exc = state["error"]
                row.update(status="error", error_type=type(exc).__name__, error=str(exc),
                           error_stage=state["error_stage"])
            else:
                started = perf_counter()
                try:
                    chunks = retriever.retrieve(case["user_input"], top_k=current.top_k)
                    row.update(status="success", retrieved_chunks=chunks,
                               retrieved_contexts=[chunk["text"] for chunk in chunks])
                except Exception as exc:
                    row.update(status="error", error_type=type(exc).__name__, error=str(exc),
                               error_stage="retrieve")
                row["latency_seconds"] = perf_counter() - started
            rows.append(row)
        errors = sum(row["status"] == "error" for row in rows)
        manifest = {**common, "setting": setting, "config": configuration,
                    "model": model, "index": index, "collection": collection,
                    "setup_seconds": state["setup_seconds"], "errors": errors,
                    "status": "error" if errors else "success"}
        if state["error"] is not None:
            manifest.update(error_stage=state["error_stage"],
                            error_type=type(state["error"]).__name__, error=str(state["error"]))
        (directory / "retrieval.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False, default=str) + "\n" for row in rows),
            encoding="utf-8")
        _write_json(directory / "manifest.json", manifest)
        summary["runs"].append({"setting": setting, "chunk_size": current.chunk_size,
                                "top_k": current.top_k, "collection": collection,
                                "path": setting, "status": manifest["status"], "errors": errors})
        summary["errors"] += errors
        summary["retrieval_count"] += len(rows)
        _write_json(destination / "manifest.json", summary)
    summary["state"] = "complete"
    _write_json(destination / "manifest.json", summary)
    return summary
