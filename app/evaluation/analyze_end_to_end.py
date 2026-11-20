#!/usr/bin/env python3
"""Análise estatística e geração de relatório para experimentos RAG vs. LLM-only.

Exemplo (a partir da raiz do projeto):
    python -m app.evaluation.analyze_end_to_end \
        --experiment-dir app/evaluation/evaluation_results/end_to_end/consolidated_99_cases_...

Dependências:
    pip install numpy pandas matplotlib scipy

O script não altera os dados originais. Cria uma pasta `analysis/` dentro do
experimento informado, contendo tabelas CSV, figuras PNG, JSON e relatório Markdown.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats
except ImportError as exc:  # pragma: no cover - mensagem de instalação
    raise SystemExit(
        "Dependência ausente. Instale com: pip install numpy pandas matplotlib scipy"
    ) from exc

DEFAULT_SETTINGS = (
    "tokens-256-topk-1",
    "tokens-256-topk-3",
    "tokens-512-topk-1",
    "tokens-512-topk-3",
)
DISPLAY_NAMES = {
    "tokens-256-topk-1": "Chunk 256 / Top-k 1",
    "tokens-256-topk-3": "Chunk 256 / Top-k 3",
    "tokens-512-topk-1": "Chunk 512 / Top-k 1",
    "tokens-512-topk-3": "Chunk 512 / Top-k 3",
    "llm_only": "LLM-only",
}
METRIC_LABELS = {
    "answer_relevancy": "Answer Relevancy",
    "answer_correctness": "Answer Correctness",
    "bleu": "BLEU",
    "rouge1": "ROUGE-1",
    "rouge2": "ROUGE-2",
    "rouge_l": "ROUGE-L",
    "faithfulness": "Faithfulness",
    "context_precision": "Context Precision",
    "context_recall": "Context Recall",
}
RAG_ONLY_METRICS = {"faithfulness", "context_precision", "context_recall"}


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Esperado objeto JSON: {path}")
    return obj


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSONL inválido em {path}:{line_no}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Registro não é objeto em {path}:{line_no}")
            records.append(value)
    return records


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def load_experiment(experiment_dir: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]], list[str]]:
    if not experiment_dir.is_dir():
        raise FileNotFoundError(f"Diretório do experimento não encontrado: {experiment_dir}")
    manifest_path = experiment_dir / "manifest.json"
    comparison_path = experiment_dir / "comparison.json"
    if not manifest_path.is_file() or not comparison_path.is_file():
        raise FileNotFoundError("O diretório precisa conter manifest.json e comparison.json")
    manifest = read_json(manifest_path)
    comparison = read_json(comparison_path)
    config_items = comparison.get("configurations") or []
    settings = [str(item.get("setting")) for item in config_items if item.get("setting")]
    if not settings:
        settings = [s for s in DEFAULT_SETTINGS if (experiment_dir / s).is_dir()]
    if len(settings) != 4 or set(settings) != set(DEFAULT_SETTINGS):
        raise ValueError(
            "O experimento precisa conter as quatro configurações esperadas: "
            + ", ".join(DEFAULT_SETTINGS)
            + f". Encontradas: {settings}"
        )
    approaches: dict[str, dict[str, Any]] = {}
    for approach in [*settings, "llm_only"]:
        base = experiment_dir / approach
        pred_path = base / "predictions.jsonl"
        score_path = base / "evaluation" / "scores.jsonl"
        if not pred_path.is_file() or not score_path.is_file():
            raise FileNotFoundError(f"Arquivos ausentes para {approach}: predictions.jsonl ou evaluation/scores.jsonl")
        predictions = read_jsonl(pred_path)
        scores = read_jsonl(score_path)
        retrieval = read_jsonl(base / "retrieval.jsonl")
        approaches[approach] = {
            "predictions": predictions,
            "scores": scores,
            "retrieval": retrieval,
        }
    metrics = comparison.get("metrics") or manifest.get("enabled_metrics") or []
    if not metrics:
        first_scores = approaches[settings[0]]["scores"]
        if first_scores:
            metrics = list((first_scores[0].get("scores") or {}).keys())
    metrics = [str(m) for m in metrics]
    if not metrics:
        raise ValueError("Não foi possível identificar as métricas habilitadas")
    return manifest, comparison, approaches, metrics


def index_records(records: list[dict[str, Any]], label: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in records:
        case_id = str(row.get("case_id") or "")
        if not case_id:
            raise ValueError(f"Registro sem case_id em {label}")
        if case_id in indexed:
            raise ValueError(f"case_id duplicado em {label}: {case_id}")
        indexed[case_id] = row
    return indexed


def validate_and_index(approaches: dict[str, dict[str, Any]]) -> tuple[list[str], dict[str, dict[str, Any]]]:
    indexed: dict[str, dict[str, Any]] = {}
    reference_ids: set[str] | None = None
    for approach, data in approaches.items():
        preds = index_records(data["predictions"], f"{approach}/predictions")
        scores = index_records(data["scores"], f"{approach}/scores")
        if set(preds) != set(scores):
            missing_scores = sorted(set(preds) - set(scores))[:5]
            missing_predictions = sorted(set(scores) - set(preds))[:5]
            raise ValueError(
                f"IDs divergentes em {approach}: sem score={missing_scores}; sem prediction={missing_predictions}"
            )
        if reference_ids is None:
            reference_ids = set(preds)
        elif set(preds) != reference_ids:
            raise ValueError(f"A abordagem {approach} não possui o mesmo conjunto de case_id das demais")
        retrieval = index_records(data["retrieval"], f"{approach}/retrieval") if data["retrieval"] else {}
        indexed[approach] = {"predictions": preds, "scores": scores, "retrieval": retrieval}
    case_ids = sorted(reference_ids or [])
    if not case_ids:
        raise ValueError("Nenhuma pergunta encontrada")
    return case_ids, indexed


def score_value(score_record: dict[str, Any], metric: str) -> float | None:
    status = (score_record.get("metric_status") or {}).get(metric)
    value = finite_number((score_record.get("scores") or {}).get(metric))
    if status is not None and status != "success":
        return None
    return value


def descriptive_statistics(indexed: dict[str, dict[str, Any]], metrics: list[str], approaches: list[str]) -> pd.DataFrame:
    rows = []
    for approach in approaches:
        score_map = indexed[approach]["scores"]
        for metric in metrics:
            vals = [v for r in score_map.values() if (v := score_value(r, metric)) is not None]
            statuses = Counter((r.get("metric_status") or {}).get(metric, "missing") for r in score_map.values())
            arr = np.asarray(vals, dtype=float)
            rows.append({
                "approach": approach,
                "approach_label": DISPLAY_NAMES.get(approach, approach),
                "metric": metric,
                "metric_label": METRIC_LABELS.get(metric, metric),
                "n_total": len(score_map),
                "n_valid": int(arr.size),
                "n_missing_or_invalid": int(len(score_map) - arr.size),
                "mean": float(np.mean(arr)) if arr.size else np.nan,
                "std": float(np.std(arr, ddof=1)) if arr.size > 1 else (0.0 if arr.size == 1 else np.nan),
                "median": float(np.median(arr)) if arr.size else np.nan,
                "q1": float(np.quantile(arr, .25)) if arr.size else np.nan,
                "q3": float(np.quantile(arr, .75)) if arr.size else np.nan,
                "min": float(np.min(arr)) if arr.size else np.nan,
                "max": float(np.max(arr)) if arr.size else np.nan,
                "status_success": statuses.get("success", 0),
                "status_error": sum(v for k, v in statuses.items() if k in {"error", "failed", "failure"}),
                "status_not_applicable": statuses.get("not_applicable", 0),
                "status_skipped": statuses.get("skipped", 0),
            })
    return pd.DataFrame(rows)


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator, iterations: int, confidence: float = .95) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan
    if len(values) == 1:
        return float(values[0]), float(values[0])
    samples = rng.choice(values, size=(iterations, len(values)), replace=True).mean(axis=1)
    alpha = 1 - confidence
    return float(np.quantile(samples, alpha / 2)), float(np.quantile(samples, 1 - alpha / 2))


def paired_test(differences: np.ndarray) -> tuple[float, float, float]:
    diffs = np.asarray(differences, dtype=float)
    diffs = diffs[np.isfinite(diffs)]
    if len(diffs) < 2:
        return np.nan, np.nan, np.nan
    mean_diff = float(np.mean(diffs))
    sd = float(np.std(diffs, ddof=1))
    effect_dz = mean_diff / sd if sd > 0 else (0.0 if mean_diff == 0 else np.inf)
    if np.allclose(diffs, 0):
        return 0.0, 1.0, effect_dz
    try:
        p = float(stats.wilcoxon(diffs, alternative="two-sided", zero_method="wilcox", method="auto").pvalue)
    except ValueError:
        p = 1.0
    return mean_diff, p, effect_dz


def holm_adjust(p_values: list[float]) -> list[float]:
    valid = [(i, p) for i, p in enumerate(p_values) if np.isfinite(p)]
    adjusted = [np.nan] * len(p_values)
    ordered = sorted(valid, key=lambda x: x[1])
    running = 0.0
    m = len(ordered)
    for rank, (idx, p) in enumerate(ordered):
        value = min(1.0, (m - rank) * p)
        running = max(running, value)
        adjusted[idx] = running
    return adjusted


def pairwise_rag_vs_baseline(indexed: dict[str, dict[str, Any]], settings: list[str], metrics: list[str], rng: np.random.Generator, iterations: int) -> pd.DataFrame:
    rows = []
    for metric in metrics:
        pending = []
        for setting in settings:
            rag_map = indexed[setting]["scores"]
            llm_map = indexed["llm_only"]["scores"]
            paired = [(cid, score_value(rag_map[cid], metric), score_value(llm_map[cid], metric)) for cid in rag_map]
            paired = [(cid, a, b) for cid, a, b in paired if a is not None and b is not None]
            if len(paired) < 2:
                pending.append({"metric": metric, "setting": setting, "n_pairs": len(paired), "mean_rag": np.nan, "mean_llm": np.nan, "mean_difference": np.nan, "ci95_low": np.nan, "ci95_high": np.nan, "p_value": np.nan, "effect_cohen_dz": np.nan, "rag_wins": 0, "ties": 0, "rag_losses": 0})
                continue
            diffs = np.array([a - b for _, a, b in paired], dtype=float)
            mean_diff, p, dz = paired_test(diffs)
            low, high = bootstrap_mean_ci(diffs, rng, iterations)
            wins = int(np.sum(diffs > 0)); losses = int(np.sum(diffs < 0)); ties = int(np.sum(np.isclose(diffs, 0)))
            pending.append({
                "metric": metric, "metric_label": METRIC_LABELS.get(metric, metric),
                "setting": setting, "setting_label": DISPLAY_NAMES[setting], "n_pairs": len(paired),
                "mean_rag": float(np.mean([a for _, a, _ in paired])),
                "mean_llm": float(np.mean([b for _, _, b in paired])),
                "mean_difference": mean_diff, "ci95_low": low, "ci95_high": high,
                "p_value": p, "effect_cohen_dz": dz,
                "rag_wins": wins, "ties": ties, "rag_losses": losses,
            })
        adjusted = holm_adjust([r["p_value"] for r in pending])
        for row, p_adj in zip(pending, adjusted):
            row["p_value_holm"] = p_adj
            rows.append(row)
    return pd.DataFrame(rows)


def rag_configuration_tests(indexed: dict[str, dict[str, Any]], settings: list[str], metrics: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    overall_rows = []
    posthoc_rows = []
    from itertools import combinations
    for metric in metrics:
        case_values = {}
        for cid in indexed[settings[0]]["scores"]:
            vals = [score_value(indexed[s]["scores"][cid], metric) for s in settings]
            if all(v is not None for v in vals):
                case_values[cid] = vals
        n = len(case_values)
        if n >= 3:
            matrix = np.array(list(case_values.values()), dtype=float)
            try:
                friedman = stats.friedmanchisquare(*[matrix[:, i] for i in range(matrix.shape[1])])
                stat, p = float(friedman.statistic), float(friedman.pvalue)
            except ValueError:
                stat, p = 0.0, 1.0
        else:
            stat, p = np.nan, np.nan
        overall_rows.append({"metric": metric, "metric_label": METRIC_LABELS.get(metric, metric), "n_complete_cases": n, "friedman_statistic": stat, "p_value": p})
        pair_pending = []
        for left, right in combinations(settings, 2):
            diffs = []
            for cid in case_values:
                a = score_value(indexed[left]["scores"][cid], metric)
                b = score_value(indexed[right]["scores"][cid], metric)
                diffs.append(a - b)
            mean_diff, p_pair, dz = paired_test(np.asarray(diffs))
            pair_pending.append({"metric": metric, "setting_a": left, "setting_b": right, "n_pairs": len(diffs), "mean_difference_a_minus_b": mean_diff, "p_value": p_pair, "effect_cohen_dz": dz})
        adjusted = holm_adjust([r["p_value"] for r in pair_pending])
        for row, p_adj in zip(pair_pending, adjusted):
            row["p_value_holm"] = p_adj
            posthoc_rows.append(row)
    friedman_adjusted = holm_adjust([row["p_value"] for row in overall_rows])
    for row, p_adj in zip(overall_rows, friedman_adjusted):
        row["p_value_holm"] = p_adj
    return pd.DataFrame(overall_rows), pd.DataFrame(posthoc_rows)


def factorial_effects(indexed: dict[str, dict[str, Any]], metrics: list[str], rng: np.random.Generator, iterations: int) -> pd.DataFrame:
    rows = []
    for metric in metrics:
        effects: dict[str, list[float]] = {"chunk_512_minus_256": [], "topk_3_minus_1": [], "interaction": []}
        for cid in indexed[DEFAULT_SETTINGS[0]]["scores"]:
            vals = {s: score_value(indexed[s]["scores"][cid], metric) for s in DEFAULT_SETTINGS}
            if any(v is None for v in vals.values()):
                continue
            # média sobre top-k: efeito de trocar chunk 256 por 512
            effects["chunk_512_minus_256"].append(((vals["tokens-512-topk-1"] + vals["tokens-512-topk-3"]) / 2) - ((vals["tokens-256-topk-1"] + vals["tokens-256-topk-3"]) / 2))
            # média sobre chunk size: efeito de trocar top-k 1 por 3
            effects["topk_3_minus_1"].append(((vals["tokens-256-topk-3"] + vals["tokens-512-topk-3"]) / 2) - ((vals["tokens-256-topk-1"] + vals["tokens-512-topk-1"]) / 2))
            # diferença-em-diferenças: interação chunk x top-k
            effects["interaction"].append((vals["tokens-512-topk-3"] - vals["tokens-512-topk-1"]) - (vals["tokens-256-topk-3"] - vals["tokens-256-topk-1"]))
        for effect_name, values in effects.items():
            arr = np.asarray(values, dtype=float)
            mean_diff, p, dz = paired_test(arr)
            low, high = bootstrap_mean_ci(arr, rng, iterations)
            rows.append({"metric": metric, "metric_label": METRIC_LABELS.get(metric, metric), "effect": effect_name, "n_complete_cases": len(arr), "mean_effect": mean_diff, "ci95_low": low, "ci95_high": high, "p_value": p, "effect_cohen_dz": dz})
        metric_rows = rows[-len(effects):]
        adjusted = holm_adjust([row["p_value"] for row in metric_rows])
        for row, p_adj in zip(metric_rows, adjusted):
            row["p_value_holm"] = p_adj
    return pd.DataFrame(rows)


def build_per_question(indexed: dict[str, dict[str, Any]], case_ids: list[str], settings: list[str], metrics: list[str]) -> pd.DataFrame:
    rows = []
    for cid in case_ids:
        question = ""
        for approach in [*settings, "llm_only"]:
            pred = indexed[approach]["predictions"].get(cid, {})
            if pred.get("user_input"):
                question = str(pred["user_input"])
                break
        row: dict[str, Any] = {"case_id": cid, "question": question}
        for approach in [*settings, "llm_only"]:
            score = indexed[approach]["scores"][cid]
            for metric in metrics:
                row[f"{approach}__{metric}"] = score_value(score, metric)
        for metric in metrics:
            baseline = row.get(f"llm_only__{metric}")
            rag_vals = {s: row.get(f"{s}__{metric}") for s in settings}
            for setting, value in rag_vals.items():
                row[f"delta_{setting}_vs_llm__{metric}"] = (value - baseline) if value is not None and baseline is not None else np.nan
            valid = {s: v for s, v in rag_vals.items() if v is not None}
            row[f"best_rag_setting__{metric}"] = max(valid, key=valid.get) if valid else None
            row[f"best_rag_score__{metric}"] = max(valid.values()) if valid else np.nan
            row[f"best_rag_minus_llm__{metric}"] = (max(valid.values()) - baseline) if valid and baseline is not None else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def retrieval_analysis(indexed: dict[str, dict[str, Any]], settings: list[str], metrics: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for setting in settings:
        data = indexed[setting]
        for cid, retrieval in data["retrieval"].items():
            chunks = retrieval.get("retrieved_chunks") or []
            contexts = retrieval.get("retrieved_contexts") or []
            token_counts = [finite_number((chunk.get("metadata") or {}).get("token_count")) for chunk in chunks if isinstance(chunk, dict)]
            token_counts = [v for v in token_counts if v is not None]
            context_chars = sum(len(str(c)) for c in contexts)
            score = data["scores"].get(cid, {})
            pred = data["predictions"].get(cid, {})
            rows.append({
                "case_id": cid, "setting": setting, "setting_label": DISPLAY_NAMES[setting],
                "retrieval_status": retrieval.get("status"),
                "retrieval_latency_seconds": finite_number(retrieval.get("latency_seconds")),
                "retrieved_chunks_count": len(chunks),
                "retrieved_contexts_count": len(contexts),
                "retrieved_tokens_sum": float(sum(token_counts)) if token_counts else np.nan,
                "retrieved_context_chars": context_chars,
                "generation_latency_seconds": finite_number(pred.get("generation_latency_seconds")),
                "answer_relevancy": score_value(score, "answer_relevancy"),
                "faithfulness": score_value(score, "faithfulness"),
                "context_recall": score_value(score, "context_recall"),
                "rouge_l": score_value(score, "rouge_l"),
            })
    retrieval_df = pd.DataFrame(rows)
    correlations = []
    if not retrieval_df.empty:
        variables = ["retrieved_chunks_count", "retrieved_tokens_sum", "retrieved_context_chars", "retrieval_latency_seconds"]
        outcomes = [m for m in ["answer_relevancy", "faithfulness", "context_recall", "rouge_l"] if m in retrieval_df.columns]
        for setting in settings:
            subset = retrieval_df[retrieval_df.setting == setting]
            for x in variables:
                for y in outcomes:
                    pairs = subset[[x, y]].dropna()
                    if len(pairs) >= 5 and pairs[x].nunique() > 1 and pairs[y].nunique() > 1:
                        result = stats.spearmanr(pairs[x], pairs[y])
                        rho, p = float(result.statistic), float(result.pvalue)
                    else:
                        rho, p = np.nan, np.nan
                    correlations.append({"setting": setting, "setting_label": DISPLAY_NAMES[setting], "variable_x": x, "metric_y": y, "n": len(pairs), "spearman_rho": rho, "p_value": p})
    return retrieval_df, pd.DataFrame(correlations)


def runtime_analysis(indexed: dict[str, dict[str, Any]], approaches: list[str], retrieval_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    latency_rows = []
    for approach in approaches:
        predictions = indexed[approach]["predictions"].values()
        retrieval_values = []
        generation_values = []
        total_values = []
        for pred in predictions:
            retrieval_latency = finite_number(pred.get("retrieval_latency_seconds"))
            generation_latency = finite_number(pred.get("generation_latency_seconds"))
            if retrieval_latency is not None:
                retrieval_values.append(retrieval_latency)
            if generation_latency is not None:
                generation_values.append(generation_latency)
            if generation_latency is not None:
                total_values.append(generation_latency + (retrieval_latency or 0.0))
        def stats_for(values: list[float], prefix: str) -> dict[str, Any]:
            arr = np.asarray(values, dtype=float)
            return {
                f"{prefix}_n": int(len(arr)),
                f"{prefix}_mean_seconds": float(np.mean(arr)) if len(arr) else np.nan,
                f"{prefix}_median_seconds": float(np.median(arr)) if len(arr) else np.nan,
                f"{prefix}_p95_seconds": float(np.quantile(arr, .95)) if len(arr) else np.nan,
            }
        row = {"approach": approach, "approach_label": DISPLAY_NAMES.get(approach, approach)}
        row.update(stats_for(retrieval_values, "retrieval"))
        row.update(stats_for(generation_values, "generation"))
        row.update(stats_for(total_values, "retrieval_plus_generation"))
        latency_rows.append(row)

    retrieval_summary_rows = []
    if not retrieval_df.empty:
        for setting, group in retrieval_df.groupby("setting", sort=False):
            row = {"setting": setting, "setting_label": DISPLAY_NAMES.get(setting, setting), "n_questions": int(len(group))}
            for column in ["retrieved_chunks_count", "retrieved_contexts_count", "retrieved_tokens_sum", "retrieved_context_chars", "retrieval_latency_seconds"]:
                values = pd.to_numeric(group[column], errors="coerce").dropna()
                row[f"{column}_mean"] = float(values.mean()) if len(values) else np.nan
                row[f"{column}_median"] = float(values.median()) if len(values) else np.nan
                row[f"{column}_p95"] = float(values.quantile(.95)) if len(values) else np.nan
            row["retrieval_success_count"] = int((group["retrieval_status"] == "success").sum())
            retrieval_summary_rows.append(row)
    return pd.DataFrame(latency_rows), pd.DataFrame(retrieval_summary_rows)


def data_quality_report(case_ids: list[str], indexed: dict[str, dict[str, Any]], approaches: list[str], metrics: list[str]) -> dict[str, Any]:
    report: dict[str, Any] = {"unique_case_ids": len(case_ids), "approaches": {}}
    for approach in approaches:
        preds = indexed[approach]["predictions"]
        scores = indexed[approach]["scores"]
        statuses = Counter(str(row.get("status", "missing")) for row in preds.values())
        generation = Counter(str(row.get("generation_status", "missing")) for row in preds.values())
        retrieval = Counter(str(row.get("retrieval_status", "missing")) for row in preds.values())
        metric_statuses = {}
        for metric in metrics:
            metric_statuses[metric] = dict(Counter((row.get("metric_status") or {}).get(metric, "missing") for row in scores.values()))
        report["approaches"][approach] = {
            "prediction_count": len(preds), "score_count": len(scores),
            "prediction_status": dict(statuses), "generation_status": dict(generation),
            "retrieval_status": dict(retrieval), "metric_status": metric_statuses,
        }
    return report


def save_figures(out_dir: Path, desc: pd.DataFrame, per_question: pd.DataFrame, retrieval_df: pd.DataFrame, metrics: list[str], settings: list[str], iterations: int, seed: int) -> list[str]:
    figures = out_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    generated: list[str] = []
    approaches = [*settings, "llm_only"]
    comparable_metrics = [m for m in metrics if m not in RAG_ONLY_METRICS]

    # 1. Médias com IC bootstrap de 95%.
    for metric in comparable_metrics:
        fig, ax = plt.subplots(figsize=(10, 5.5))
        labels, means, lows, highs = [], [], [], []
        for approach in approaches:
            row = desc[(desc.approach == approach) & (desc.metric == metric)]
            if row.empty or pd.isna(row.iloc[0]["mean"]):
                continue
            # IC bootstrap aproximado reconstruído apenas com estatísticas não é válido;
            # usamos média +/- erro padrão como visualização descritiva e identificamos no título.
            n = int(row.iloc[0].n_valid); mean = float(row.iloc[0]["mean"]); std = float(row.iloc[0]["std"])
            margin = 1.96 * std / math.sqrt(n) if n > 1 and math.isfinite(std) else 0.0
            labels.append(DISPLAY_NAMES[approach]); means.append(mean); lows.append(margin); highs.append(margin)
        if labels:
            x = np.arange(len(labels))
            ax.bar(x, means, yerr=[lows, highs], capsize=4)
            ax.set_xticks(x, labels, rotation=18, ha="right")
            ax.set_ylabel("Pontuação média")
            ax.set_title(f"{METRIC_LABELS.get(metric, metric)} — média ± 1,96 erro-padrão")
            ax.grid(axis="y", alpha=.25)
            fig.tight_layout()
            filename = f"metric_comparison_{metric}.png"; fig.savefig(figures / filename, dpi=180); plt.close(fig); generated.append(filename)

    # 2. Distribuições por configuração (boxplot).
    for metric in comparable_metrics:
        cols = [f"{a}__{metric}" for a in approaches if f"{a}__{metric}" in per_question]
        data = [per_question[c].dropna().to_numpy(dtype=float) for c in cols]
        if not any(len(v) for v in data):
            continue
        fig, ax = plt.subplots(figsize=(10, 5.5))
        ax.boxplot(data, showfliers=True)
        ax.set_xticks(range(1, len(cols) + 1), [DISPLAY_NAMES[c.removesuffix(f"__{metric}")] for c in cols])
        ax.set_ylabel("Pontuação"); ax.set_title(f"Distribuição por pergunta — {METRIC_LABELS.get(metric, metric)}")
        ax.tick_params(axis="x", rotation=18); ax.grid(axis="y", alpha=.25); fig.tight_layout()
        filename = f"metric_distributions_{metric}.png"; fig.savefig(figures / filename, dpi=180); plt.close(fig); generated.append(filename)

    # 3. Heatmap das médias das métricas comparáveis.
    if comparable_metrics:
        matrix = []
        for metric in comparable_metrics:
            matrix.append([desc[(desc.approach == a) & (desc.metric == metric)]["mean"].iloc[0] for a in approaches])
        fig, ax = plt.subplots(figsize=(10, max(4, .7 * len(comparable_metrics))))
        im = ax.imshow(np.asarray(matrix, dtype=float), aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(range(len(approaches)), [DISPLAY_NAMES[a] for a in approaches], rotation=20, ha="right")
        ax.set_yticks(range(len(comparable_metrics)), [METRIC_LABELS.get(m, m) for m in comparable_metrics])
        for i, row in enumerate(matrix):
            for j, val in enumerate(row):
                if np.isfinite(val): ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, label="Média"); ax.set_title("Médias por métrica e abordagem")
        fig.tight_layout(); filename = "metric_heatmap.png"; fig.savefig(figures / filename, dpi=180); plt.close(fig); generated.append(filename)

    # 4. Interação chunk size x top-k.
    interaction_metrics = comparable_metrics[:]
    if interaction_metrics:
        fig, axes = plt.subplots(1, len(interaction_metrics), figsize=(5 * len(interaction_metrics), 4.5), squeeze=False)
        for ax, metric in zip(axes[0], interaction_metrics):
            vals_256 = [per_question[f"tokens-256-topk-{k}__{metric}"].mean() for k in (1, 3)]
            vals_512 = [per_question[f"tokens-512-topk-{k}__{metric}"].mean() for k in (1, 3)]
            ax.plot([1, 3], vals_256, marker="o", label="Chunk 256")
            ax.plot([1, 3], vals_512, marker="o", label="Chunk 512")
            ax.set_xticks([1, 3]); ax.set_xlabel("Top-k"); ax.set_ylabel("Média")
            ax.set_title(METRIC_LABELS.get(metric, metric)); ax.grid(alpha=.25)
        axes[0][0].legend(); fig.suptitle("Interação entre chunk size e top-k")
        fig.tight_layout(); filename = "interaction_effects.png"; fig.savefig(figures / filename, dpi=180); plt.close(fig); generated.append(filename)

    # 5. Heatmap por pergunta (Answer Relevancy, ou primeira métrica comparável).
    heat_metric = "answer_relevancy" if "answer_relevancy" in comparable_metrics else (comparable_metrics[0] if comparable_metrics else None)
    if heat_metric:
        matrix = per_question[[f"{a}__{heat_metric}" for a in approaches]].to_numpy(dtype=float)
        fig, ax = plt.subplots(figsize=(10, max(7, len(per_question) * .075)))
        im = ax.imshow(matrix, aspect="auto", interpolation="nearest", vmin=0, vmax=1)
        ax.set_xticks(range(len(approaches)), [DISPLAY_NAMES[a] for a in approaches], rotation=20, ha="right")
        ax.set_ylabel("Pergunta (ordem por case_id)"); ax.set_title(f"Pontuação por pergunta — {METRIC_LABELS.get(heat_metric, heat_metric)}")
        fig.colorbar(im, ax=ax, label="Pontuação"); fig.tight_layout()
        filename = f"per_question_heatmap_{heat_metric}.png"; fig.savefig(figures / filename, dpi=180); plt.close(fig); generated.append(filename)

    # 6. Contextos recuperados e latência por configuração.
    if not retrieval_df.empty:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        data = [retrieval_df.loc[retrieval_df.setting == s, "retrieved_chunks_count"].dropna().to_numpy() for s in settings]
        axes[0].boxplot(data); axes[0].set_xticks(range(1, len(settings) + 1), [DISPLAY_NAMES[s] for s in settings]); axes[0].set_title("Chunks recuperados"); axes[0].tick_params(axis="x", rotation=20)
        lat = [retrieval_df.loc[retrieval_df.setting == s, "retrieval_latency_seconds"].dropna().to_numpy() for s in settings]
        axes[1].boxplot(lat); axes[1].set_xticks(range(1, len(settings) + 1), [DISPLAY_NAMES[s] for s in settings]); axes[1].set_title("Latência de retrieval (s)"); axes[1].tick_params(axis="x", rotation=20)
        for ax in axes: ax.grid(axis="y", alpha=.25)
        fig.tight_layout(); filename = "retrieval_contexts_and_latency.png"; fig.savefig(figures / filename, dpi=180); plt.close(fig); generated.append(filename)
    return generated


def fmt(value: Any, digits: int = 4) -> str:
    number = finite_number(value)
    return "—" if number is None else f"{number:.{digits}f}"


def generate_report(path: Path, manifest: dict[str, Any], case_count: int, desc: pd.DataFrame, rag_vs_llm: pd.DataFrame, friedman: pd.DataFrame, posthoc: pd.DataFrame, factorial: pd.DataFrame, retrieval_df: pd.DataFrame, correlations: pd.DataFrame, latency_df: pd.DataFrame, retrieval_summary_df: pd.DataFrame, quality: dict[str, Any], figures: list[str], metrics: list[str]) -> None:
    lines = [
        f"# Relatório de análise — {path.parent.parent.name}", "",
        f"- **Data da análise (UTC):** {datetime.now(timezone.utc).isoformat()}",
        f"- **Perguntas analisadas:** {case_count}",
        f"- **Modelo de geração:** {manifest.get('generation_model', 'não informado')}",
        f"- **Métricas:** {', '.join(metrics)}", "",
        "> Este relatório descreve associações e diferenças observadas. Resultados estatísticos não demonstram causalidade. BLEU/ROUGE medem sobreposição lexical com a referência, não equivalência semântica.", "",
        "## 1. Qualidade e integridade dos dados", "",
        f"Foram analisadas {case_count} perguntas, com validação de correspondência dos `case_id` entre as quatro configurações RAG e o baseline LLM-only. Consulte `data_quality.json` para os status por abordagem e métrica.", "",
        "## 2. Estatísticas descritivas", "",
        "A tabela completa está em `descriptive_statistics.csv`. Valores de média e dispersão devem ser interpretados junto com `n_valid`, pois falhas ou métricas não aplicáveis podem reduzir a amostra efetiva.", "",
        "| Abordagem | Métrica | N válido | Média | DP | Mediana | Q1–Q3 |", "|---|---|---:|---:|---:|---:|---:|",
    ]
    for _, r in desc.iterrows():
        lines.append(f"| {r.approach_label} | {r.metric_label} | {int(r.n_valid)} | {fmt(r['mean'])} | {fmt(r['std'])} | {fmt(r['median'])} | {fmt(r.q1)}–{fmt(r.q3)} |")
    lines += ["", "## 3. RAG versus LLM-only", "", "Diferenças positivas indicam pontuação maior para RAG. Os testes de Wilcoxon são pareados por pergunta; `p_value_holm` corrige as quatro comparações de configurações RAG contra o baseline dentro de cada métrica.", "", "| Métrica | Configuração | N pares | Δ média | IC 95% | p ajustado | Vitórias/empates/derrotas |", "|---|---|---:|---:|---:|---:|---:|"]
    for _, r in rag_vs_llm.iterrows():
        ci = f"[{fmt(r.ci95_low)}, {fmt(r.ci95_high)}]"
        wtl = f"{int(r.rag_wins)}/{int(r.ties)}/{int(r.rag_losses)}"
        lines.append(f"| {r.get('metric_label', r.metric)} | {r.get('setting_label', r.setting)} | {int(r.n_pairs)} | {fmt(r.mean_difference)} | {ci} | {fmt(r.p_value_holm)} | {wtl} |")
    lines += ["", "## 4. Comparação entre configurações RAG", "", "O teste de Friedman avalia diferenças globais entre as quatro configurações para as perguntas com pontuação válida em todas elas. Os testes posteriores de Wilcoxon são pareados e corrigidos por Holm.", "", "| Métrica | N completo | Friedman χ² | p | p ajustado (Holm) |", "|---|---:|---:|---:|---:|"]
    for _, r in friedman.iterrows(): lines.append(f"| {r.metric_label} | {int(r.n_complete_cases)} | {fmt(r.friedman_statistic, 3)} | {fmt(r.p_value)} | {fmt(r.p_value_holm)} |")
    lines += ["", "Os resultados pareados posteriores estão em `rag_configuration_posthoc.csv`.", "", "## 5. Efeitos fatoriais: chunk size e top-k", "", "Os efeitos são contrastes dentro das mesmas perguntas. Interação positiva significa que o efeito de top-k=3 versus top-k=1 foi mais positivo com chunk 512 do que com chunk 256.", "", "| Métrica | Efeito | N | Estimativa | IC 95% | p | p ajustado (Holm) |", "|---|---|---:|---:|---:|---:|---:|"]
    for _, r in factorial.iterrows(): lines.append(f"| {r.metric_label} | {r.effect} | {int(r.n_complete_cases)} | {fmt(r.mean_effect)} | [{fmt(r.ci95_low)}, {fmt(r.ci95_high)}] | {fmt(r.p_value)} | {fmt(r.p_value_holm)} |")
    lines += ["", "## 6. Recuperação de contexto", "", f"Foram encontrados {len(retrieval_df)} registros de retrieval nas quatro configurações. Estatísticas detalhadas por pergunta estão em `retrieval_analysis.csv`; correlações exploratórias de Spearman estão em `retrieval_correlations.csv`.", "", "Correlações não demonstram que a quantidade de contexto ou a latência cause alterações na qualidade. A análise também depende de os campos de chunks e latência estarem registrados nos JSONL.", "", "## 7. Análise por pergunta", "", "`per_question_analysis.csv` contém pontuações de cada abordagem, deltas contra LLM-only e melhor configuração RAG por métrica. Use-o para selecionar exemplos de melhoria, regressão e divergência para discussão qualitativa.", "", "## 8. Figuras geradas", ""]
    for fig in figures: lines.append(f"- `figures/{fig}`")
    lines += ["", "## 9. Limitações de interpretação", "", "- Métricas automáticas são proxies e não substituem avaliação humana de correção factual e utilidade.",
              "- Answer Relevancy, Faithfulness e métricas RAGAS dependem do avaliador baseado em LLM; seus resultados podem conter variabilidade e vieses do avaliador.",
              "- BLEU e ROUGE privilegiam sobreposição textual e podem penalizar respostas corretas formuladas de outra maneira.",
              "- Testes estatísticos pressupõem que as unidades pareadas (perguntas) sejam representativas; o tamanho e a seleção do conjunto limitam a generalização.",
              "- Os testes de múltiplas comparações são corrigidos por Holm nos blocos explicitamente indicados; análises exploratórias devem ser apresentadas como tais.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment-dir", required=True, type=Path, help="Pasta do experimento consolidado (ou execução completa com as quatro configurações).")
    parser.add_argument("--output-dir", type=Path, default=None, help="Destino da análise. Padrão: <experiment-dir>/analysis.")
    parser.add_argument("--bootstrap-iterations", type=int, default=5000, help="Número de reamostragens bootstrap (padrão: 5000).")
    parser.add_argument("--seed", type=int, default=42, help="Semente para reprodutibilidade.")
    args = parser.parse_args()
    if args.bootstrap_iterations < 100:
        parser.error("--bootstrap-iterations deve ser >= 100")

    experiment_dir = args.experiment_dir.expanduser().resolve()
    output_dir = (args.output_dir.expanduser().resolve() if args.output_dir else experiment_dir / "analysis")
    if output_dir == experiment_dir or experiment_dir in output_dir.parents and output_dir.name != "analysis":
        raise SystemExit("O diretório de saída deve ser separado dos dados originais (use a pasta analysis).")

    try:
        manifest, comparison, approaches_data, metrics = load_experiment(experiment_dir)
        case_ids, indexed = validate_and_index(approaches_data)
        declared_cases = manifest.get("cases_count")
        if declared_cases is not None and int(declared_cases) != len(case_ids):
            raise ValueError(f"manifest.json declara {declared_cases} casos, mas foram encontrados {len(case_ids)} case_id únicos")
        settings = [str(item["setting"]) for item in comparison.get("configurations", [])] or list(DEFAULT_SETTINGS)
        approaches = [*settings, "llm_only"]
        rng = np.random.default_rng(args.seed)

        desc = descriptive_statistics(indexed, metrics, approaches)
        comparable_metrics = [m for m in metrics if m not in RAG_ONLY_METRICS]
        rag_vs_llm = pairwise_rag_vs_baseline(indexed, settings, comparable_metrics, rng, args.bootstrap_iterations)
        friedman, posthoc = rag_configuration_tests(indexed, settings, metrics)
        factorial = factorial_effects(indexed, metrics, rng, args.bootstrap_iterations)
        per_question = build_per_question(indexed, case_ids, settings, metrics)
        retrieval_df, correlations = retrieval_analysis(indexed, settings, metrics)
        latency_df, retrieval_summary_df = runtime_analysis(indexed, approaches, retrieval_df)
        quality = data_quality_report(case_ids, indexed, approaches, metrics)

        output_dir.mkdir(parents=True, exist_ok=True)
        desc.to_csv(output_dir / "descriptive_statistics.csv", index=False)
        rag_vs_llm.to_csv(output_dir / "rag_vs_llm.csv", index=False)
        friedman.to_csv(output_dir / "statistical_tests_friedman.csv", index=False)
        posthoc.to_csv(output_dir / "rag_configuration_posthoc.csv", index=False)
        factorial.to_csv(output_dir / "factorial_effects.csv", index=False)
        per_question.to_csv(output_dir / "per_question_analysis.csv", index=False)
        retrieval_df.to_csv(output_dir / "retrieval_analysis.csv", index=False)
        correlations.to_csv(output_dir / "retrieval_correlations.csv", index=False)
        latency_df.to_csv(output_dir / "latency_analysis.csv", index=False)
        retrieval_summary_df.to_csv(output_dir / "retrieval_summary.csv", index=False)
        write_json(output_dir / "data_quality.json", quality)

        figures = save_figures(output_dir, desc, per_question, retrieval_df, metrics, settings, args.bootstrap_iterations, args.seed)
        generate_report(output_dir / "report.md", manifest, len(case_ids), desc, rag_vs_llm, friedman, posthoc, factorial, retrieval_df, correlations, latency_df, retrieval_summary_df, quality, figures, metrics)

        summary = {
            "experiment_dir": str(experiment_dir), "analysis_dir": str(output_dir),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "case_count": len(case_ids), "settings": settings, "metrics": metrics,
            "bootstrap_iterations": args.bootstrap_iterations, "seed": args.seed,
            "files": [
                "summary.json", "data_quality.json", "descriptive_statistics.csv", "rag_vs_llm.csv",
                "statistical_tests_friedman.csv", "rag_configuration_posthoc.csv", "factorial_effects.csv",
                "per_question_analysis.csv", "retrieval_analysis.csv", "retrieval_summary.csv", "retrieval_correlations.csv", "latency_analysis.csv", "report.md",
            ], "figures": figures,
        }
        write_json(output_dir / "summary.json", json_safe(summary))
        print("\nAnálise concluída com sucesso.")
        print(f"Perguntas: {len(case_ids)}")
        print(f"Abordagens: {len(approaches)}")
        print(f"Métricas: {', '.join(metrics)}")
        print(f"Resultados: {output_dir}")
        print(f"Relatório: {output_dir / 'report.md'}")
    except Exception as exc:
        raise SystemExit(f"Falha na análise: {exc}") from exc


if __name__ == "__main__":
    main()
