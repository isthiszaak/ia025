"""End-to-end experiment: retrieval, RAG/LLM-only generation and evaluation."""

import argparse
import asyncio
from dataclasses import replace
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import time
import unicodedata

from dotenv import load_dotenv
from importlib.metadata import version
from langchain_core.prompts import PromptTemplate

from app.backend.llm import get_llm
from app.backend.prompts import RAG_PROMPT
from app.evaluation.ragas_adapter import (
    EvaluatorConfig,
    build_metrics,
    evaluate_predictions,
)
from app.retrieval.config import RetrievalConfig
from app.retrieval.experiments import run_comparisons


ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env", override=False)

DEFAULT_DATASET_PATH = (
    ROOT / "data/retrieval/raw/"
    "9f0f4263928c6152a40f888da3e63375a24bb0f3/"
    "questions_QA_2024_v1.1.json"
)


def load_cases(path):
    cases = []
    with Path(path).open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                cases.append(json.loads(line))
    return cases


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            default=str,
            allow_nan=False,
        ) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as target:
        for record in records:
            target.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    default=str,
                    allow_nan=False,
                ) + "\n"
            )


def _response_text(response):
    if hasattr(response, "content"):
        response = response.content
    return str(response)


def generate_answer(question, contexts, chain):
    context_text = "\n\n---\n\n".join(contexts)
    response = chain.invoke({
        "user_input": question,
        "context": context_text,
    })
    return _response_text(response)


def generate_llm_only_answer(question, llm):
    """Generate a baseline answer from the question without retrieved context."""
    response = llm.invoke(
        "Answer the following question using your general knowledge. "
        "Do not assume that any external documents or retrieved context "
        "have been provided.\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )
    return _response_text(response)


def normalize_question(text):
    text = unicodedata.normalize("NFKD", text)
    text = "".join(
        char for char in text
        if not unicodedata.combining(char)
    )
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def enrich_cases_with_references(cases, dataset_path):
    with Path(dataset_path).open(encoding="utf-8") as file:
        dataset = json.load(file)

    references = {
        normalize_question(item["question_text"]): item
        for item in dataset
    }

    enriched = []
    for case in cases:
        match = references.get(normalize_question(case["user_input"]))
        enriched_case = case.copy()

        if match:
            answer = match["answer_cleaned"]
            if isinstance(answer, list):
                answer = "\n".join(str(part) for part in answer)
            enriched_case["reference"] = str(answer)
            enriched_case["reference_match"] = "exact"
        else:
            enriched_case["reference"] = None
            enriched_case["reference_match"] = "not_found"

        enriched.append(enriched_case)

    return enriched


def generate_rag_predictions(retrieval_path, predictions_path, llm):
    template = PromptTemplate(
        input_variables=["context", "user_input"],
        template=RAG_PROMPT,
    )
    chain = template | llm
    predictions = []

    with Path(retrieval_path).open(encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            row = json.loads(line)
            contexts = row.get("retrieved_contexts", [])
            if not isinstance(contexts, list):
                contexts = []

            record = {
                "approach": "rag",
                "case_id": row.get("case_id"),
                "user_input": row.get("user_input"),
                "reference": row.get("reference"),
                "retrieved_contexts": contexts,
                "retrieval_status": row.get("status", "error"),
                "generation_status": "error",
                "response": None,
                "retrieval_latency_seconds": row.get("latency_seconds"),
                "generation_latency_seconds": None,
                "status": "error",
            }

            if row.get("status") != "success":
                record["error"] = (
                    "Generation skipped because retrieval failed: "
                    + str(row.get("error", "Unknown retrieval error"))
                )
                predictions.append(record)
                continue

            try:
                started = time.perf_counter()
                response = generate_answer(
                    row["user_input"], contexts, chain
                )
                record.update(
                    response=response,
                    generation_status="success",
                    generation_latency_seconds=round(
                        time.perf_counter() - started, 6
                    ),
                    status="success",
                )
            except Exception as exc:
                record.update(
                    generation_status="error",
                    status="error",
                    error_type=type(exc).__name__,
                    error=str(exc),
                )

            predictions.append(record)

    write_jsonl(predictions_path, predictions)
    return predictions


def generate_llm_only_predictions(cases, predictions_path, llm):
    """Generate each baseline answer exactly once, independent of retrieval."""
    predictions = []

    for index, case in enumerate(cases, start=1):
        record = {
            "approach": "llm_only",
            "case_id": case.get("case_id"),
            "user_input": case.get("user_input"),
            "reference": case.get("reference"),
            "retrieved_contexts": [],
            "retrieval_status": "not_applicable",
            "generation_status": "error",
            "response": None,
            "retrieval_latency_seconds": None,
            "generation_latency_seconds": None,
            "status": "error",
        }

        try:
            started = time.perf_counter()
            response = generate_llm_only_answer(case["user_input"], llm)
            record.update(
                response=response,
                generation_status="success",
                generation_latency_seconds=round(
                    time.perf_counter() - started, 6
                ),
                status="success",
            )
        except Exception as exc:
            record.update(
                error_type=type(exc).__name__,
                error=str(exc),
            )

        predictions.append(record)
        print(
            f"LLM-only {index}/{len(cases)} | "
            f"case={record['case_id']} | "
            f"status={record['generation_status']}",
            flush=True,
        )

    write_jsonl(predictions_path, predictions)
    return predictions


async def evaluate_all_settings(run_dir, config):
    metrics, clients = build_metrics(config)
    comparison = []
    ragas_version = version("ragas")

    try:
        for setting_dir in sorted(run_dir.iterdir()):
            if not setting_dir.is_dir() or setting_dir.name == "llm_only":
                continue

            predictions_path = setting_dir / "predictions.jsonl"
            if not predictions_path.exists():
                continue

            evaluation_dir = setting_dir / "evaluation"
            await evaluate_predictions(
                predictions_path=predictions_path,
                output_dir=evaluation_dir,
                metrics=metrics,
                config=config,
                ragas_version=ragas_version,
            )

            summary = json.loads(
                (evaluation_dir / "summary.json").read_text(
                    encoding="utf-8"
                )
            )

            match = re.fullmatch(
                r"tokens-(\d+)-topk-(\d+)", setting_dir.name
            )
            if not match:
                raise ValueError(
                    f"Unexpected retrieval setting directory: {setting_dir.name}"
                )

            comparison.append({
                "setting": setting_dir.name,
                "chunk_size": int(match.group(1)),
                "top_k": int(match.group(2)),
                "metrics": summary["metrics"],
            })

        llm_only_dir = run_dir / "llm_only"
        llm_only_predictions = llm_only_dir / "predictions.jsonl"
        if not llm_only_predictions.exists():
            raise FileNotFoundError(
                f"LLM-only predictions not found: {llm_only_predictions}"
            )

        llm_only_evaluation_dir = llm_only_dir / "evaluation"
        await evaluate_predictions(
            predictions_path=llm_only_predictions,
            output_dir=llm_only_evaluation_dir,
            metrics=metrics,
            config=config,
            ragas_version=ragas_version,
        )
        llm_only_summary = json.loads(
            (llm_only_evaluation_dir / "summary.json").read_text(
                encoding="utf-8"
            )
        )

    finally:
        for client in clients:
            await client.close()

    return comparison, {
        "setting": "llm_only",
        "metrics": llm_only_summary["metrics"],
        "total_cases": llm_only_summary["total_cases"],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run end-to-end RAG and LLM-only evaluation"
    )
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunk-overlap", type=int, default=50)
    parser.add_argument("--fixed-chunk-size", type=int, default=512)
    parser.add_argument("--fixed-top-k", type=int, default=3)
    parser.add_argument("--selection", type=str, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--build-indexes", action="store_true")
    parser.add_argument(
        "--all-combinations",
        action="store_true",
        help="Execute all CHUNK_SIZES x TOP_K_VALUES combinations",
    )
    parser.add_argument(
        "--reference-dataset",
        type=Path,
        default=Path(
            os.getenv(
                "EVAL_REFERENCE_DATASET_PATH",
                str(DEFAULT_DATASET_PATH),
            )
        ),
        help="JSON dataset used to enrich cases with reference answers",
    )

    args = parser.parse_args()
    cases = load_cases(args.cases)
    cases = enrich_cases_with_references(cases, args.reference_dataset)

    for case in cases:
        print(
            case.get("case_id"),
            case.get("reference_match"),
            bool(case.get("reference")),
        )

    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be greater than zero")
        cases = cases[:args.limit]

    if not cases:
        raise ValueError("No evaluation cases were provided")

    selection = (
        None if args.selection.lower() == "all"
        else Path(args.selection)
    )

    retrieval_config = replace(
        RetrievalConfig.from_env(),
        chunk_unit="tokens",
        chunk_overlap=args.chunk_overlap,
        corpus_selection=selection,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    retrieval_summary = run_comparisons(
        config=retrieval_config,
        cases=cases,
        output_dir=args.output_dir,
        fixed_top_k=args.fixed_top_k,
        fixed_chunk_size=args.fixed_chunk_size,
        build=args.build_indexes,
        all_combinations=args.all_combinations,
    )

    run_dir = Path(retrieval_summary["output_dir"])
    llm = get_llm()

    # Generate the LLM-only baseline once, not once per retrieval setting.
    llm_only_dir = run_dir / "llm_only"
    generate_llm_only_predictions(
        cases,
        llm_only_dir / "predictions.jsonl",
        llm,
    )

    # Generate RAG answers separately for every retrieval configuration.
    for run in retrieval_summary["runs"]:
        setting_dir = run_dir / run["setting"]
        generate_rag_predictions(
            setting_dir / "retrieval.jsonl",
            setting_dir / "predictions.jsonl",
            llm,
        )

    evaluator_config = EvaluatorConfig.from_env()
    rag_comparison, llm_only_comparison = asyncio.run(
        evaluate_all_settings(run_dir, evaluator_config)
    )

    comparison_payload = {
        "run_id": retrieval_summary["run_id"],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "end_to_end_rag_vs_llm_only",
        "metrics": list(evaluator_config.selected_metrics),
        "configurations": rag_comparison,
        "llm_only": llm_only_comparison,
    }

    comparison_path = run_dir / "comparison.json"
    write_json(comparison_path, comparison_payload)

    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        stage="end_to_end_rag_vs_llm_only",
        generation_model=getattr(llm, "model", "configured_llm"),
        ragas_version=version("ragas"),
        enabled_metrics=list(evaluator_config.selected_metrics),
        comparison_path=str(comparison_path),
        llm_only_predictions_path=str(
            (llm_only_dir / "predictions.jsonl").resolve()
        ),
    )
    write_json(manifest_path, manifest)

    print(json.dumps({
        "run_id": retrieval_summary["run_id"],
        "output_dir": str(run_dir),
        "comparison_path": str(comparison_path),
        "enabled_metrics": list(evaluator_config.selected_metrics),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
