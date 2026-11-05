"""Evaluate saved predictions with configurable RAGAS and lexical metrics."""

import argparse
import asyncio
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import re
import statistics
import time

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

ALL_METRICS = (
    "answer_relevancy",
    "answer_correctness",
    "bleu",
    "rouge1",
    "rouge2",
    "rouge_l",
    "faithfulness",
    "context_precision",
    "context_recall",
)

DEFAULT_METRICS = (
    "answer_relevancy",
    "bleu",
    "rouge1",
    "rouge2",
    "rouge_l",
    "faithfulness",
    "context_recall",
)

RAGAS_METRICS = {
    "answer_relevancy",
    "answer_correctness",
    "faithfulness",
    "context_precision",
    "context_recall",
}

RESPONSE_METRICS = {
    "answer_relevancy",
    "answer_correctness",
    "bleu",
    "rouge1",
    "rouge2",
    "rouge_l",
    "faithfulness",
}

METRIC_LABELS = {
    "answer_relevancy": "Answer Relevancy",
    "answer_correctness": "Answer Correctness",
    "bleu": "BLEU-4",
    "rouge1": "ROUGE-1",
    "rouge2": "ROUGE-2",
    "rouge_l": "ROUGE-L",
    "faithfulness": "Faithfulness",
    "context_precision": "Context Precision",
    "context_recall": "Context Recall",
}


def parse_metrics(value):
    """Parse and validate the comma-separated EVAL_METRICS value."""
    if value is None or not value.strip():
        selected = list(DEFAULT_METRICS)
    else:
        selected = [item.strip().lower() for item in value.split(",") if item.strip()]

    unknown = sorted(set(selected) - set(ALL_METRICS))
    if unknown:
        raise ValueError(
            "Unknown EVAL_METRICS value(s): "
            + ", ".join(unknown)
            + ". Available metrics: "
            + ", ".join(ALL_METRICS)
        )

    duplicates = sorted(
        name for name, count in Counter(selected).items() if count > 1
    )
    if duplicates:
        raise ValueError(
            "Duplicate metric(s) in EVAL_METRICS: " + ", ".join(duplicates)
        )

    if not selected:
        raise ValueError("EVAL_METRICS must select at least one metric")

    return tuple(selected)


@dataclass(frozen=True)
class EvaluatorConfig:
    llm_model: str = ""
    embedding_model: str = ""
    llm_base_url: str = "http://localhost:11434/v1"
    embedding_base_url: str = "http://localhost:11434/v1"
    temperature: float = 0.0
    timeout_seconds: float = 120.0
    max_tokens: int = 2048
    top_p: float = 0.1
    strictness: int = 3
    selected_metrics: tuple = DEFAULT_METRICS

    def __post_init__(self):
        if not self.selected_metrics:
            raise ValueError("At least one evaluation metric must be selected")

        unknown = set(self.selected_metrics) - set(ALL_METRICS)
        if unknown:
            raise ValueError(f"Unsupported metrics: {sorted(unknown)}")

        if RAGAS_METRICS.intersection(self.selected_metrics):
            if not self.llm_model.strip():
                raise ValueError("Set EVAL_LLM_MODEL for the selected RAGAS metrics")

        if (
            {"answer_relevancy", "answer_correctness"}
            .intersection(self.selected_metrics)
            and not self.embedding_model.strip()
        ):
            raise ValueError(
                "Set EVAL_EMBEDDING_MODEL for Answer Relevancy/Correctness"
            )

        if (
            not math.isfinite(self.temperature)
            or not math.isfinite(self.timeout_seconds)
            or not math.isfinite(self.top_p)
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
            top_p=float(os.getenv("EVAL_LLM_TOP_P", "0.1")),
            strictness=int(os.getenv("EVAL_STRICTNESS", "3")),
            selected_metrics=parse_metrics(os.getenv("EVAL_METRICS")),
        )


def build_metrics(config):
    """Build only the selected RAGAS metrics; lexical metrics are local."""
    selected = set(config.selected_metrics)
    requested_ragas = selected.intersection(RAGAS_METRICS)
    if not requested_ragas:
        return {}, ()

    from openai import AsyncOpenAI
    from ragas.llms import llm_factory
    import ragas.metrics.collections as collection_metrics

    llm_client = AsyncOpenAI(
        base_url=config.llm_base_url,
        api_key=os.getenv("EVAL_LLM_API_KEY", "ollama"),
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

    embedding_client = None
    embeddings = None
    if selected.intersection({"answer_relevancy", "answer_correctness"}):
        from ragas.embeddings.base import embedding_factory

        embedding_client = AsyncOpenAI(
            base_url=config.embedding_base_url,
            api_key=os.getenv("EVAL_EMBEDDING_API_KEY", "ollama"),
            timeout=config.timeout_seconds,
            max_retries=0,
        )
        embeddings = embedding_factory(
            "openai",
            model=config.embedding_model,
            client=embedding_client,
            interface="modern",
        )

    class_names = {
        "answer_relevancy": "AnswerRelevancy",
        "answer_correctness": "AnswerCorrectness",
        "faithfulness": "Faithfulness",
        "context_precision": "ContextPrecision",
        "context_recall": "ContextRecall",
    }

    metrics = {}
    for metric_name in requested_ragas:
        class_name = class_names[metric_name]
        metric_class = getattr(collection_metrics, class_name, None)
        if metric_class is None:
            raise ImportError(
                f"Installed RAGAS version does not expose "
                f"{class_name} in ragas.metrics.collections"
            )

        if metric_name == "answer_relevancy":
            metric = metric_class(
                llm=llm,
                embeddings=embeddings,
                strictness=config.strictness,
            )
        elif metric_name == "answer_correctness":
            metric = metric_class(llm=llm, embeddings=embeddings)
        else:
            metric = metric_class(llm=llm)

        metrics[metric_name] = metric

    clients = tuple(
        client for client in (llm_client, embedding_client)
        if client is not None
    )
    return metrics, clients


def _tokens(text):
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def _ngrams(tokens, n):
    return Counter(
        tuple(tokens[index:index + n])
        for index in range(max(0, len(tokens) - n + 1))
    )


def sentence_bleu(reference, hypothesis, max_order=4):
    """Smoothed sentence BLEU-4 with brevity penalty (0..1)."""
    reference_tokens = _tokens(reference)
    hypothesis_tokens = _tokens(hypothesis)

    if not hypothesis_tokens or not reference_tokens:
        return 0.0

    precisions = []
    for n in range(1, max_order + 1):
        hyp_ngrams = _ngrams(hypothesis_tokens, n)
        ref_ngrams = _ngrams(reference_tokens, n)
        total = sum(hyp_ngrams.values())
        if total == 0:
            precisions.append(1.0)
            continue
        clipped = sum(
            min(count, ref_ngrams[gram])
            for gram, count in hyp_ngrams.items()
        )
        # Add-one smoothing prevents short sentences from collapsing to zero.
        precisions.append((clipped + 1) / (total + 1))

    brevity_penalty = (
        1.0
        if len(hypothesis_tokens) >= len(reference_tokens)
        else math.exp(1 - len(reference_tokens) / len(hypothesis_tokens))
    )
    return brevity_penalty * math.exp(
        sum(math.log(value) for value in precisions) / max_order
    )


def rouge_n_f1(reference, hypothesis, n):
    reference_ngrams = _ngrams(_tokens(reference), n)
    hypothesis_ngrams = _ngrams(_tokens(hypothesis), n)

    ref_total = sum(reference_ngrams.values())
    hyp_total = sum(hypothesis_ngrams.values())
    if ref_total == 0 or hyp_total == 0:
        return 0.0

    overlap = sum(
        min(count, hypothesis_ngrams[gram])
        for gram, count in reference_ngrams.items()
    )
    precision = overlap / hyp_total
    recall = overlap / ref_total
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def rouge_l_f1(reference, hypothesis):
    reference_tokens = _tokens(reference)
    hypothesis_tokens = _tokens(hypothesis)
    if not reference_tokens or not hypothesis_tokens:
        return 0.0

    # Dynamic-programming LCS length, using O(min(n, m)) memory.
    if len(hypothesis_tokens) > len(reference_tokens):
        reference_tokens, hypothesis_tokens = hypothesis_tokens, reference_tokens

    previous = [0] * (len(hypothesis_tokens) + 1)
    for ref_token in reference_tokens:
        current = [0]
        for index, hyp_token in enumerate(hypothesis_tokens, start=1):
            if ref_token == hyp_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[index - 1]))
        previous = current

    lcs = previous[-1]
    precision = lcs / len(hypothesis_tokens)
    recall = lcs / len(reference_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _deterministic_score(name, reference, response):
    if name == "bleu":
        return sentence_bleu(reference, response)
    if name == "rouge1":
        return rouge_n_f1(reference, response, 1)
    if name == "rouge2":
        return rouge_n_f1(reference, response, 2)
    if name == "rouge_l":
        return rouge_l_f1(reference, response)
    raise ValueError(f"Unsupported deterministic metric: {name}")


async def score_case(case, metrics, config):
    selected_metrics = config.selected_metrics
    scores = {name: None for name in selected_metrics}
    statuses = {name: "not_applicable" for name in selected_metrics}
    reasons = {}
    metric_durations = {name: None for name in selected_metrics}

    if not isinstance(case, dict):
        raise ValueError("Each prediction must be a JSON object")

    record = {
        "case_id": case.get("case_id"),
        "user_input": case.get("user_input"),
        "approach": case.get("approach", "rag"),
        "scores": scores,
        "metric_status": statuses,
        "reasons": reasons,
        "metric_duration_seconds": metric_durations,
    }

    reference = case.get("reference")
    has_reference = isinstance(reference, str) and bool(reference.strip())
    response = case.get("response")
    has_response = isinstance(response, str) and bool(response.strip())

    contexts = case.get("retrieved_contexts", [])
    if not isinstance(contexts, list):
        contexts = []
    contexts = [
        context for context in contexts
        if isinstance(context, str) and context.strip()
    ]
    has_contexts = bool(contexts)

    applicable = {
        "answer_relevancy": True,
        "answer_correctness": has_reference,
        "bleu": has_reference,
        "rouge1": has_reference,
        "rouge2": has_reference,
        "rouge_l": has_reference,
        "faithfulness": has_contexts,
        "context_precision": has_reference and has_contexts,
        "context_recall": has_reference and has_contexts,
    }

    for name in selected_metrics:
        if not applicable[name]:
            if name in {"answer_correctness", "bleu", "rouge1", "rouge2", "rouge_l"} and not has_reference:
                reasons[name] = "missing_reference"
            elif name in {"faithfulness", "context_precision", "context_recall"} and not has_contexts:
                reasons[name] = "missing_retrieved_contexts"
            continue

        if name in {"context_precision", "context_recall"} and not has_reference:
            reasons[name] = "missing_reference"
            continue

        if name in RESPONSE_METRICS and not has_response:
            statuses[name] = (
                "skipped"
                if case.get("generation_status") == "error"
                else "error"
            )
            reasons[name] = (
                "generation_failed"
                if case.get("generation_status") == "error"
                else "missing_response"
            )
            continue

        metric_start = time.perf_counter()
        try:
            if name in {"bleu", "rouge1", "rouge2", "rouge_l"}:
                value = _deterministic_score(name, reference, response)
            else:
                kwargs = {"user_input": case.get("user_input")}

                if name in {
                    "answer_relevancy", "answer_correctness", "faithfulness"
                }:
                    kwargs["response"] = response

                if name in {
                    "faithfulness", "context_precision", "context_recall"
                }:
                    kwargs["retrieved_contexts"] = contexts

                if name in {
                    "answer_correctness", "context_precision", "context_recall"
                }:
                    kwargs["reference"] = reference

                if (
                    not isinstance(kwargs.get("user_input"), str)
                    or not kwargs["user_input"].strip()
                ):
                    raise ValueError("user_input must be nonempty")

                result = await asyncio.wait_for(
                    metrics[name].ascore(**kwargs),
                    timeout=config.timeout_seconds,
                )
                value = float(result.value)

            if not math.isfinite(value):
                raise ValueError("Metric returned a nonfinite score")

            scores[name] = float(value)
            statuses[name] = "success"

        except Exception as exc:
            statuses[name] = "error"
            last_completion = getattr(exc, "last_completion", None)
            reasons[name] = (
                f"{type(exc).__name__}: {repr(exc)} | "
                f"last_completion={repr(last_completion)}"
            )
        finally:
            elapsed = time.perf_counter() - metric_start
            metric_durations[name] = round(elapsed, 3)
            print(
                f"    {name} | Status: {statuses[name]} "
                f"| Duration: {elapsed:.2f}s",
                flush=True,
            )

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
    selected_metrics = config.selected_metrics

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

    values = {name: [] for name in selected_metrics}
    metric_times = {name: [] for name in selected_metrics}
    counts = {
        name: {
            "valid_count": 0,
            "failure_count": 0,
            "not_applicable_count": 0,
            "skipped_count": 0,
        }
        for name in selected_metrics
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    total_cases = sum(
        1 for line in predictions_path.open(encoding="utf-8") if line.strip()
    )
    if total_cases == 0:
        raise ValueError(f"No predictions found in {predictions_path}")

    with predictions_path.open(encoding="utf-8") as source:
        with (output_dir / "scores.jsonl").open("x", encoding="utf-8") as target:
            case_index = 0
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue

                case_index += 1
                print(
                    f"\nCase {case_index}/{total_cases} "
                    f"| Progress: {case_index / total_cases * 100:.1f}%",
                    flush=True,
                )

                try:
                    case = json.loads(line)
                    record = await score_case(case, metrics, config)
                except Exception as exc:
                    record = {
                        "case_id": None,
                        "approach": None,
                        "scores": {name: None for name in selected_metrics},
                        "metric_status": {name: "error" for name in selected_metrics},
                        "reasons": {},
                        "metric_duration_seconds": {
                            name: None for name in selected_metrics
                        },
                        "error": f"{type(exc).__name__}: {exc}",
                    }

                record["prediction_line_number"] = line_number

                for name in selected_metrics:
                    elapsed = record["metric_duration_seconds"].get(name)
                    if elapsed is not None:
                        metric_times[name].append(elapsed)

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
                    json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
                )
                target.flush()

    summary = {
        "total_cases": total_cases,
        "standard_deviation": "population",
        "selected_metrics": list(selected_metrics),
        "metrics": {
            name: {
                "mean": statistics.mean(values[name]) if values[name] else None,
                "stddev": statistics.pstdev(values[name]) if values[name] else None,
                "total_duration_seconds": round(sum(metric_times[name]), 3),
                "mean_duration_seconds": (
                    round(statistics.mean(metric_times[name]), 3)
                    if metric_times[name] else None
                ),
                "attempted_count": len(metric_times[name]),
                **counts[name],
            }
            for name in selected_metrics
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
        "enabled_metrics": list(selected_metrics),
        "total_cases": total_cases,
    }

    for filename, payload in (
        ("summary.json", summary),
        ("manifest.json", manifest),
    ):
        (output_dir / filename).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
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
        description="Evaluate saved predictions with configurable metrics"
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

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    config = EvaluatorConfig.from_env()
    output_dir = asyncio.run(_run(args, config))
    print(f"Evaluation saved to {output_dir}")


if __name__ == "__main__":
    main()
