"""PowerShell-friendly commands: prepare, build, query."""

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter
from uuid import uuid4

from .config import RetrievalConfig
from .data import DEFAULT_REVISION, prepare_dataset
from .index import Retriever, build_index, dependency_versions


def _print(value):
    # JSON escapes keep console/piped output valid on legacy Windows encodings.
    # Saved JSONL files remain UTF-8 with their original Unicode text.
    print(json.dumps(value, ensure_ascii=True, indent=2, default=str))


def _parser():
    parser = argparse.ArgumentParser(description="BR-TaxQA: preparação e recuperação, sem geração")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "build", "query", "compare"):
        command = sub.add_parser(name)
        command.add_argument("--data-dir", type=Path)
        if name in {"prepare", "build"}:
            command.add_argument("--revision", default=DEFAULT_REVISION)
            command.add_argument("--raw-dir", type=Path, help="JSON locais já baixados, validados pelo cache")
            command.add_argument("--include-carf", action="store_true", help="Inclui corpus documental CARF, sem criar gabaritos")
        if name == "build":
            command.add_argument("--use-prepared", action="store_true", help="Usa exatamente o corpus já preparado")
        if name in {"build", "query", "compare"}:
            command.add_argument("--index-dir", type=Path)
            command.add_argument("--selection", help="JSON com document_ids, ou all para todo o corpus")
            command.add_argument("--chunk-size", type=int)
            command.add_argument("--chunk-overlap", type=int)
            command.add_argument("--chunk-unit", choices=("characters", "tokens"))
            command.add_argument("--tokenizer-id")
            command.add_argument("--tokenizer-revision")
        if name in {"query", "compare"}:
            if name == "query":
                command.add_argument("question", nargs="?")
            command.add_argument("--top-k", type=int)
            command.add_argument("--cases", type=Path, help="JSONL com case_id e user_input")
            command.add_argument("--limit", type=int, help="Máximo de casos, na ordem original")
            command.add_argument("--output-dir", type=Path, help="Cria subpasta de execução sem sobrescrever")
        if name == "compare":
            command.add_argument("--build-indexes", action="store_true", help="Constrói/retoma os três índices do corpus preparado")
    return parser


def _cases(args):
    if bool(args.question) == bool(args.cases):
        raise ValueError("Informe uma pergunta OU --cases")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit deve ser positivo")
    if args.question:
        return [{"case_id": "interactive", "user_input": args.question}]
    cases = []
    with args.cases.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            if not isinstance(item, dict) or not isinstance(item.get("user_input"), str) or not item["user_input"].strip():
                raise ValueError(f"Caso inválido na linha {line_number}")
            cases.append({**item, "case_id": item.get("case_id", f"line_{line_number}")})
            if args.limit is not None and len(cases) >= args.limit:
                break
    if not cases:
        raise ValueError("Arquivo sem perguntas")
    return cases


def _query(args, config):
    from .experiments import corpus_snapshot
    cases = _cases(args)
    retriever = None
    opening_error = None
    setup_started = perf_counter()
    try:
        retriever = Retriever(config)
    except Exception as exc:
        opening_error = exc
    setup_seconds = perf_counter() - setup_started
    rows = []
    for case in cases:
        started = perf_counter()
        # References remain metadata; only user_input is sent to the embedder.
        row = {k: case[k] for k in ("case_id", "user_input", "reference", "references",
                                    "all_formatted_references", "all_formated_references") if k in case}
        try:
            if opening_error is not None:
                raise opening_error
            chunks = retriever.retrieve(case["user_input"])
            row.update(status="success", retrieved_chunks=chunks,
                       retrieved_contexts=[chunk["text"] for chunk in chunks])
        except Exception as exc:
            row.update(status="error", error_type=type(exc).__name__, error=str(exc),
                       error_stage="open" if opening_error is not None else "retrieve",
                       retrieved_chunks=[], retrieved_contexts=[])
        row.update(latency_seconds=None if opening_error is not None else perf_counter() - started,
                   collection=retriever.collection_name if retriever else None,
                   top_k=config.top_k, config=_serializable_config(config))
        rows.append(row)
    manifest = {"run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid4().hex[:8],
                "timestamp_utc": datetime.now(timezone.utc).isoformat(), "stage": "retrieval_only",
                "config": asdict(config), "index": retriever.manifest if retriever else None,
                "corpus": corpus_snapshot(config),
                "dependencies": dependency_versions(), "cases_count": len(rows),
                "setup_seconds": setup_seconds,
                "errors": sum(row["status"] == "error" for row in rows)}
    if args.cases:
        manifest.update(cases_path=str(args.cases.resolve()),
                        cases_sha256=hashlib.sha256(args.cases.read_bytes()).hexdigest(), limit=args.limit)
    if args.output_dir:
        destination = args.output_dir / manifest["run_id"]
        destination.mkdir(parents=True, exist_ok=False)
        (destination / "retrieval.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        (destination / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        _print({"output_dir": destination, "cases_count": len(rows), "errors": manifest["errors"]})
    elif args.cases:
        for row in rows:
            print(json.dumps(row, ensure_ascii=True))
    else:
        _print(rows[0])
    return 1 if manifest["errors"] else 0


def _serializable_config(config):
    return json.loads(json.dumps(asdict(config), default=str))


def main(argv=None):
    args = _parser().parse_args(argv)
    try:
        config = RetrievalConfig.from_env()
        overrides = {name: getattr(args, name) for name in
                     ("data_dir", "index_dir", "chunk_size", "chunk_overlap", "top_k",
                      "chunk_unit", "tokenizer_id", "tokenizer_revision")
                     if getattr(args, name, None) is not None}
        config = replace(config, **overrides)
        if getattr(args, "selection", None) is not None:
            config = replace(config, corpus_selection=None if args.selection.lower() == "all" else Path(args.selection))
        if args.command in {"prepare", "build"}:
            if getattr(args, "use_prepared", False):
                if args.include_carf or args.raw_dir is not None:
                    raise ValueError("--use-prepared não pode ser combinado com opções de preparação")
                dataset = None
            else:
                dataset = prepare_dataset(config.data_dir, revision=args.revision, raw_dir=args.raw_dir,
                                          include_carf=args.include_carf)
            _print(dataset if args.command == "prepare" else build_index(config))
            return 0
        if args.command == "compare":
            from .experiments import run_comparisons
            if args.cases is None or args.output_dir is None:
                raise ValueError("compare exige --cases e --output-dir")
            args.question = None
            # Baselines are explicit; CLI/environment top-k must not enlarge the design.
            result = run_comparisons(config, _cases(args), args.output_dir,
                                     fixed_top_k=args.top_k if args.top_k is not None else 3,
                                     fixed_chunk_size=args.chunk_size if args.chunk_size is not None else 512,
                                     build=args.build_indexes)
            _print(result)
            return 1 if result["errors"] else 0
        return _query(args, config)
    except (Exception, KeyboardInterrupt) as exc:
        print(f"retrieval: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 130 if isinstance(exc, KeyboardInterrupt) else 1


if __name__ == "__main__":
    raise SystemExit(main())
