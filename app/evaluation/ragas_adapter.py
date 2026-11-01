
"""Score saved predictions using RAGAS collections metrics."""

import argparse
import asyncio
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import statistics


METRICS = (
    "answer_relevancy",
    "answer_correctness",
    "faithfulness",
    "context_precision",
    "context_recall",
)


@dataclass(frozen=True)
class EvaluatorConfig:
    llm_model: str
    embedding_model: str
    llm_base_url: str = "http://localhost:11434/v1"
    embedding_base_url: str = "http://localhost:11434/v1"
    temperature: float = 0.0
    timeout_seconds: float = 120.0
    max_tokens: int = 2048
    top_p: float = 0.1
    strictness: int = 3

    def __post_init__(self):
        if not self.llm_model.strip() or not self.embedding_model.strip():
            raise ValueError(
                "Set EVAL_LLM_MODEL and EVAL_EMBEDDING_MODEL explicitly"
            )
        if (
            not math.isfinite(self.temperature)
            or not math.isfinite(self.timeout_seconds)
        ):
            raise ValueError("Evaluator values must be finite")
        if (
            self.timeout_seconds <= 0
            or self.max_tokens <= 0
            or self.strictness <= 0
        ):
            raise ValueError("Evaluator limits must be positive")

    @classmethod
    def from_env(cls):
        return cls(
            llm_model=os.getenv("EVAL_LLM_MODEL", ""),
            embedding_model=os.getenv("EVAL_EMBEDDING_MODEL", ""),
            llm_base_url=os.getenv(
                "EVAL_LLM_BASE_URL", "http://localhost:11434/v1"
            ),
            embedding_base_url=os.getenv(
                "EVAL_EMBEDDING_BASE_URL", "http://localhost:11434/v1"
            ),
            temperature=float(os.getenv("EVAL_LLM_TEMPERATURE", "0")),
            timeout_seconds=float(os.getenv("EVAL_TIMEOUT_SECONDS", "120")),
            max_tokens=int(os.getenv("EVAL_LLM_MAX_TOKENS", "2048")),
        )


def build_metrics(config):
    from openai import AsyncOpenAI
    from ragas.embeddings.base import embedding_factory
    from ragas.llms import llm_factory
    from ragas.metrics.collections import (
        AnswerCorrectness,
        AnswerRelevancy,
        ContextPrecision,
        ContextRecall,
        Faithfulness,
    )

    llm_client = AsyncOpenAI(
        base_url=config.llm_base_url,
        api_key=os.getenv("EVAL_LLM_API_KEY", "ollama"),
        timeout=config.timeout_seconds,
        max_retries=0,
    )

    embedding_client = AsyncOpenAI(
        base_url=config.embedding_base_url,
        api_key=os.getenv("EVAL_EMBEDDING_API_KEY", "ollama"),
        timeout=config.timeout_seconds,
        max_retries=0,
    )

    llm = llm_factory(
        config.llm_model,
        client=llm_client,
        provider="openai",
        adapter="instructor",
        temperature=config.temperature,
        top_p=config.top_p,
        max_tokens=config.max_tokens,
    )

    embeddings = embedding_factory(
        "openai",
        model=config.embedding_model,
        client=embedding_client,
        interface="modern",
    )

    metrics = {
        "answer_relevancy": AnswerRelevancy(
            llm=llm,
            embeddings=embeddings,
            strictness=config.strictness,
        ),
        "answer_correctness": AnswerCorrectness(
            llm=llm,
            embeddings=embeddings,
        ),
        "faithfulness": Faithfulness(llm=llm),
        "context_precision": ContextPrecision(llm=llm),
        "context_recall": ContextRecall(llm=llm),
    }

    return metrics, (llm_client, embedding_client)


async def score_case(case, metrics, timeout_seconds):
    scores = {name: None for name in METRICS}
    statuses = {name: "not_applicable" for name in METRICS}
    reasons = {}

    record = {
        "scores": scores,
        "metric_status": statuses,
        "reasons": reasons,
    }

    if not isinstance(case, dict):
        raise ValueError("Each prediction must be a JSON object")

    record.update(
        case_id=case.get("case_id"),
        user_input=case.get("user_input"),
    )

    reference = case.get("reference")
    has_reference = isinstance(reference, str) and bool(reference.strip())

    contexts = case.get("retrieved_contexts", [])
    if not isinstance(contexts, list):
        contexts = []

    contexts = [
        context for context in contexts
        if isinstance(context, str) and context.strip()
    ]

    has_contexts = bool(contexts)

    # Define which metrics are applicable to this prediction.
    applicable = {
        "answer_relevancy": True,
        "answer_correctness": has_reference,
        "faithfulness": has_contexts,
        "context_precision": has_reference and has_contexts,
        "context_recall": has_reference and has_contexts,
    }

    if not has_reference:
        reasons["answer_correctness"] = "missing_reference"
        reasons["context_precision"] = "missing_reference"
        reasons["context_recall"] = "missing_reference"

    if not has_contexts:
        reasons["faithfulness"] = "missing_retrieved_contexts"
        if has_reference:
            reasons["context_precision"] = "missing_retrieved_contexts"
            reasons["context_recall"] = "missing_retrieved_contexts"

    # Generation-dependent metrics cannot run after generation failure.
    generation_failed = case.get("generation_status") == "error"

    if generation_failed:
        for name in (
            "answer_relevancy",
            "answer_correctness",
            "faithfulness",
        ):
            if applicable[name]:
                statuses[name] = "skipped"
                reasons[name] = "generation_failed"

    for name, is_applicable in applicable.items():
        if not is_applicable or statuses[name] == "skipped":
            continue

        # Retrieval metrics can still run if generation failed.
        if name in ("answer_relevancy", "answer_correctness", "faithfulness"):
            if generation_failed:
                continue

        try:
            kwargs = {
                "user_input": case.get("user_input"),
            }

            if name in (
                "answer_relevancy",
                "answer_correctness",
                "faithfulness",
            ):
                kwargs["response"] = case.get("response")

            if name in (
                "faithfulness",
                "context_precision",
                "context_recall",
            ):
                kwargs["retrieved_contexts"] = contexts

            if name in (
                "answer_correctness",
                "context_precision",
                "context_recall",
            ):
                kwargs["reference"] = reference

            for field in ("user_input",):
                if (
                    not isinstance(kwargs.get(field), str)
                    or not kwargs[field].strip()
                ):
                    raise ValueError(f"{field} must be nonempty")

            if "response" in kwargs:
                if (
                    not isinstance(kwargs["response"], str)
                    or not kwargs["response"].strip()
                ):
                    raise ValueError("response must be nonempty")

            result = await asyncio.wait_for(
                metrics[name].ascore(**kwargs),
                timeout=timeout_seconds,
            )

            value = float(result.value)

            if not math.isfinite(value):
                raise ValueError("Metric returned a nonfinite score")

            scores[name] = value
            statuses[name] = "success"

        except Exception as exc:
            statuses[name] = "error"
            reasons[name] = f"{type(exc).__name__}: {exc}"

    return record


async def evaluate_predictions(
    predictions_path,
    output_dir,
    metrics,
    config,
    ragas_version,
):
    predictions_path = Path(predictions_path)
    output_dir = Path(output_dir)

    source_manifest_path = predictions_path.parent / "manifest.json"

    manifest = (
        json.loads(source_manifest_path.read_text(encoding="utf-8"))
        if source_manifest_path.exists()
        else {}
    )

    if not isinstance(manifest, dict):
        raise ValueError("manifest.json must contain a JSON object")

    if any(
        (output_dir / name).exists()
        for name in ("scores.jsonl", "summary.json")
    ):
        raise FileExistsError(
            "Evaluation outputs already exist; choose another --output-dir"
        )

    values = {name: [] for name in METRICS}

    counts = {
        name: {
            "valid_count": 0,
            "failure_count": 0,
            "not_applicable_count": 0,
            "skipped_count": 0,
        }
        for name in METRICS
    }

    total = 0
    output_dir.mkdir(parents=True, exist_ok=True)

    with predictions_path.open(encoding="utf-8") as source:
        with (output_dir / "scores.jsonl").open(
            "x", encoding="utf-8"
        ) as target:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue

                total += 1

                try:
                    case = json.loads(line)
                    record = await score_case(
                        case, metrics, config.timeout_seconds
                    )

                except Exception as exc:
                    record = {
                        "case_id": None,
                        "scores": {
                            name: None for name in METRICS
                        },
                        "metric_status": {
                            name: "error" if name in metrics
                            else "not_applicable"
                            for name in METRICS
                        },
                        "reasons": {},
                        "error": f"{type(exc).__name__}: {exc}",
                    }

                record["prediction_line_number"] = line_number

                for name in METRICS:
                    status = record["metric_status"][name]

                    count_key = {
                        "success": "valid_count",
                        "error": "failure_count",
                        "not_applicable": "not_applicable_count",
                        "skipped": "skipped_count",
                    }[status]

                    counts[name][count_key] += 1

                    if status == "success":
                        values[name].append(record["scores"][name])

                target.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        allow_nan=False,
                    ) + "\n"
                )
                target.flush()

    summary = {
        "total_cases": total,
        "standard_deviation": "population",
        "metrics": {
            name: {
                "mean": (
                    statistics.mean(values[name])
                    if values[name] else None
                ),
                "stddev": (
                    statistics.pstdev(values[name])
                    if values[name] else None
                ),
                **counts[name],
            }
            for name in METRICS
        },
    }

    manifest["evaluation"] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "predictions_path": str(predictions_path.resolve()),
        "output_dir": str(output_dir.resolve()),
        "ragas_version": ragas_version,
        "api": "ragas.metrics.collections",
        "transport": "openai_compatible",
        "structured_output": "instructor_json",
        "config": asdict(config),
        "enabled_metrics": list(metrics),
        "answer_correctness_weights": [0.75, 0.25],
        "total_cases": total,
    }

    for filename, payload in (
        ("summary.json", summary),
        ("manifest.json", manifest),
    ):
        (output_dir / filename).write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            ) + "\n",
            encoding="utf-8",
        )

    return output_dir


async def _run(args, config):
    metrics, clients = build_metrics(config)

    try:
        return await evaluate_predictions(
            args.predictions,
            args.output_dir or args.predictions.parent,
            metrics,
            config,
            version("ragas"),
        )
    finally:
        for client in clients:
            await client.close()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate saved predictions with RAGAS"
    )

    parser.add_argument(
        "predictions",
        type=Path,
        help="Path to an existing predictions.jsonl",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Defaults to the predictions directory",
    )

    args = parser.parse_args()

    from dotenv import load_dotenv

    load_dotenv(
        Path(__file__).resolve().parents[2] / ".env"
    )

    config = EvaluatorConfig.from_env()
    output_dir = asyncio.run(_run(args, config))

    print(f"Evaluation saved to {output_dir}")


if __name__ == "__main__":
    main()
