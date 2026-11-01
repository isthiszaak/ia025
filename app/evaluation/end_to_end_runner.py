
"""End-to-end RAG evaluation: retrieval, contextual generation and RAGAS."""

import argparse
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
from dataclasses import replace

from dotenv import load_dotenv

from app.backend.llm import get_llm
from app.retrieval.config import RetrievalConfig
from app.retrieval.experiments import run_comparisons
from app.evaluation.ragas_adapter import (
    EvaluatorConfig,
    build_metrics,
    evaluate_predictions,
)

from importlib.metadata import version
from langchain_core.prompts import PromptTemplate


ROOT = Path(__file__).resolve().parents[2]

RAG_PROMPT = """You are a helpful assistant.
Answer the question using the provided context.

If the context does not contain enough information to answer,
clearly state that the available information is insufficient.
Do not invent facts.

Retrieved context:
{context}

User question:
{user_input}

Answer:"""


def load_cases(path):
    cases = []

    with Path(path).open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                cases.append(json.loads(line))

    return cases


def write_json(path, payload):
    Path(path).write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            default=str,
        ) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path, records):
    with Path(path).open("w", encoding="utf-8") as target:
        for record in records:
            target.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    default=str,
                ) + "\n"
            )


def generate_answer(question, contexts, chain):
    context_text = "\n\n---\n\n".join(contexts)

    response = chain.invoke({
        "user_input": question,
        "context": context_text,
    })

    if hasattr(response, "content"):
        response = response.content

    return str(response)


def generate_predictions(retrieval_path, predictions_path, llm):
    template = PromptTemplate(
        input_variables=["context", "user_input"],
        template=RAG_PROMPT,
    )

    chain = template | llm

    retrieval_rows = []

    with Path(retrieval_path).open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                retrieval_rows.append(json.loads(line))

    predictions = []

    for row in retrieval_rows:
        record = {
            "case_id": row.get("case_id"),
            "user_input": row.get("user_input"),
            "reference": row.get("reference"),
            "retrieved_contexts": row.get("retrieved_contexts", []),
            "retrieval_status": row.get("status", "error"),
            "generation_status": "error",
            "response": None,
            "retrieval_latency_seconds": row.get("latency_seconds"),
            "generation_latency_seconds": None,
        }

        if row.get("status") != "success":
            record["error"] = (
                "Generation skipped because retrieval failed: "
                + str(row.get("error", "Unknown retrieval error"))
            )
            record["status"] = "error"
            predictions.append(record)
            continue

        try:
            import time

            started = time.perf_counter()

            response = generate_answer(
                row["user_input"],
                record["retrieved_contexts"],
                chain,
            )

            record.update(
                response=response,
                generation_status="success",
                generation_latency_seconds=time.perf_counter() - started,
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


async def evaluate_all_settings(run_dir, config):
    metrics, clients = build_metrics(config)
    comparison = []

    try:
        for setting_dir in sorted(run_dir.iterdir()):
            if not setting_dir.is_dir():
                continue

            predictions_path = setting_dir / "predictions.jsonl"

            if not predictions_path.exists():
                continue

            ragas_dir = setting_dir / "ragas"

            await evaluate_predictions(
                predictions_path=predictions_path,
                output_dir=ragas_dir,
                metrics=metrics,
                config=config,
                ragas_version=version("ragas"),
            )

            summary_path = ragas_dir / "summary.json"
            summary = json.loads(
                summary_path.read_text(encoding="utf-8")
            )

            comparison.append({
                "setting": setting_dir.name,
                "chunk_size": int(
                    setting_dir.name.split("-")[1]
                ),
                "top_k": int(
                    setting_dir.name.split("-")[-1]
                ),
                "metrics": summary["metrics"],
            })

    finally:
        for client in clients:
            await client.close()

    return comparison


def main():
    parser = argparse.ArgumentParser(
        description="Run end-to-end RAG evaluation"
    )

    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunk-overlap", type=int, default=50)
    parser.add_argument("--fixed-chunk-size", type=int, default=512)
    parser.add_argument("--fixed-top-k", type=int, default=3)
    parser.add_argument("--selection", type=str, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--build-indexes", action="store_true")

    args = parser.parse_args()

    load_dotenv(ROOT / ".env", override=False)

    cases = load_cases(args.cases)

    if args.limit is not None:
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

    # Stage 1: execute the five retrieval configurations.
    retrieval_summary = run_comparisons(
        config=retrieval_config,
        cases=cases,
        output_dir=args.output_dir,
        fixed_top_k=args.fixed_top_k,
        fixed_chunk_size=args.fixed_chunk_size,
        build=args.build_indexes,
    )

    run_dir = Path(retrieval_summary["output_dir"])

    # Stage 2: generate answers using each configuration's contexts.
    llm = get_llm()

    for run in retrieval_summary["runs"]:
        setting_dir = run_dir / run["setting"]

        retrieval_path = setting_dir / "retrieval.jsonl"
        predictions_path = setting_dir / "predictions.jsonl"

        generate_predictions(
            retrieval_path,
            predictions_path,
            llm,
        )

    # Stage 3: evaluate all predictions with RAGAS.
    evaluator_config = EvaluatorConfig.from_env()

    comparison = asyncio.run(
        evaluate_all_settings(run_dir, evaluator_config)
    )

    # Stage 4: save consolidated comparison.
    comparison_payload = {
        "run_id": retrieval_summary["run_id"],
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "end_to_end_rag",
        "metrics": [
            "answer_relevancy",
            "answer_correctness",
            "faithfulness",
            "context_precision",
            "context_recall",
        ],
        "configurations": comparison,
    }

    write_json(
        run_dir / "comparison.json",
        comparison_payload,
    )

    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    manifest.update(
        stage="end_to_end_rag",
        generation_model=getattr(llm, "model", "configured_llm"),
        ragas_version=version("ragas"),
        comparison_path=str(run_dir / "comparison.json"),
    )

    write_json(manifest_path, manifest)

    print(json.dumps({
        "run_id": retrieval_summary["run_id"],
        "output_dir": str(run_dir),
        "comparison_path": str(run_dir / "comparison.json"),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
