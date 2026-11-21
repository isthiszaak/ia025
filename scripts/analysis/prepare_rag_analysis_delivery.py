"""Prepare versionable tables and figures for the RAG configuration analysis.

Requires the existing analyzer and audit_rag_configurations.py outputs.
Never changes experiment inputs or runs language models.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import shutil
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from app.evaluation.analyze_end_to_end import DEFAULT_SETTINGS, holm_adjust, score_value


def jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def failure(setting, cid, stage, metric, reason):
    text = str(reason or "")
    return {"setting": setting, "case_id": cid, "stage": stage, "metric": metric,
            "reason_excerpt": text[:450],
            "reason_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, default=ROOT / "files_ia_025/consolidated_120_cases_20261108T165307Z")
    parser.add_argument("--analysis-dir", type=Path, default=ROOT / "evaluation_results/rag_configuration_analysis")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs/rag_configuration_analysis")
    args = parser.parse_args()
    source, analysis, out = args.experiment_dir.resolve(), args.analysis_dir.resolve(), args.output_dir.resolve()
    tables, figures = out / "tables", out / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(exist_ok=True)
    audit = json.loads((analysis / "audit/audit.json").read_text())
    for name, digest in audit["input_sha256"].items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Input changed since audit: {name}")

    exports = {
        "audit/factorial_multiplicity_sensitivity.csv": "factorial_effects.csv",
        "audit/friedman_permutation.csv": "friedman.csv",
        "audit/posthoc_multiplicity_sensitivity.csv": "posthoc.csv",
        "audit/quality_same_complete_cohort.csv": "quality_paired.csv",
        "audit/retrieval_correlations_adjusted.csv": "retrieval_correlations.csv",
        "audit/latency_tail_diagnostics.csv": "latency_diagnostics.csv",
        "reproduced/retrieval_summary.csv": "retrieval_summary.csv",
        "reproduced/latency_analysis.csv": "latency_summary.csv",
    }
    for original, name in exports.items():
        shutil.copyfile(analysis / original, tables / name)
    for original in sorted((analysis / "audit").glob("interaction_*.png")):
        shutil.copyfile(original, figures / original.name)

    rows, sensitivities, failures, metric_statuses = [], [], [], []
    predictions, scores = {}, {}
    for setting in DEFAULT_SETTINGS:
        predictions[setting] = {r["case_id"]: r for r in jsonl(source / setting / "predictions.jsonl")}
        scores[setting] = {r["case_id"]: r for r in jsonl(source / setting / "evaluation/scores.jsonl")}
        retrieval = {r["case_id"]: r for r in jsonl(source / setting / "retrieval.jsonl")}
        assert set(predictions[setting]) == set(scores[setting]) == set(retrieval)
        paired = {r["case_id"]: r for r in jsonl(source / setting / "pairs.jsonl")}
        for cid, pred in predictions[setting].items():
            response = pred.get("response") or ""
            words = response.split()[-2000:]
            grams = [tuple(words[i:i + 4]) for i in range(max(0, len(words) - 3))]
            repetition = 1 - len(set(grams)) / len(grams) if grams else np.nan
            rows.append({
                "setting": setting, "case_id": cid, "status": pred.get("status"),
                "response_characters": len(response),
                "generation_seconds": pred.get("generation_latency_seconds"),
                "retrieval_seconds": retrieval[cid].get("latency_seconds"),
                "tail_4gram_repetition_fraction": repetition,
                "pair_completed_at_utc": paired.get(cid, {}).get("completed_at_utc"),
                "response_prefix": response[:140].replace("\n", " "),
                "response_suffix": response[-200:].replace("\n", " "),
            })
            if pred.get("status") != "success":
                failures.append(failure(setting, cid, "generation", "", pred.get("error")))
        metric_names = next(iter(scores[setting].values()))["scores"]
        for metric in metric_names:
            counts = Counter(r.get("metric_status", {}).get(metric, "missing") for r in scores[setting].values())
            metric_statuses.append({"setting": setting, "metric": metric, **{s: counts[s] for s in ("success", "error", "skipped", "not_applicable", "missing")}})
            for cid, record in scores[setting].items():
                if record["metric_status"].get(metric) == "error":
                    failures.append(failure(setting, cid, "evaluation", metric, record.get("reasons", {}).get(metric)))

    frame = pd.DataFrame(rows)
    frame.to_csv(tables / "response_diagnostics.csv", index=False)
    pd.DataFrame(failures).to_csv(tables / "failures.csv", index=False)
    pd.DataFrame(metric_statuses).to_csv(tables / "metric_status_counts.csv", index=False)
    diagnostic_rows, generation_correlations = [], []
    for setting, group in frame.groupby("setting", sort=False):
        valid = group.dropna(subset=["generation_seconds"])
        for threshold in (5000, 10000, 20000):
            large = valid[valid.response_characters > threshold]
            sensitivities.append({
                "setting": setting, "threshold_characters": threshold,
                "n_long": len(large), "n_timed": len(valid),
                "fraction_generation_time_long": large.generation_seconds.sum() / valid.generation_seconds.sum(),
                "median_seconds_remaining_diagnostic_only": valid[valid.response_characters <= threshold].generation_seconds.median(),
            })
        long = valid[valid.response_characters > 10000]
        diagnostic_rows.append({
            "setting": setting, "n_long": len(long),
            "n_long_with_tail_4gram_repetition_above_80pct": int((long.tail_4gram_repetition_fraction > .8).sum()),
            "median_tail_repetition_long": long.tail_4gram_repetition_fraction.median(),
            "median_response_characters_all": valid.response_characters.median(),
        })
        result = stats.spearmanr(valid.response_characters, valid.generation_seconds)
        generation_correlations.append({"setting": setting, "n": len(valid), "rho": result.statistic, "p_approximate": result.pvalue})
    correlations = pd.DataFrame(generation_correlations)
    correlations["p_holm_4_settings"] = holm_adjust(correlations.p_approximate.tolist())
    correlations.to_csv(tables / "response_length_latency_correlations.csv", index=False)
    pd.DataFrame(diagnostic_rows).to_csv(tables / "repetition_summary.csv", index=False)
    pd.DataFrame(sensitivities).to_csv(tables / "latency_threshold_sensitivity.csv", index=False)

    # Comparison of latency and quality on precisely the same cohort per panel.
    tradeoff_rows = []
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.7))
    labels = ["256/k=1", "256/k=3", "512/k=1", "512/k=3"]
    for ax, metric in zip(axes, ("answer_relevancy", "faithfulness", "context_recall")):
        cohort = [cid for cid in predictions[DEFAULT_SETTINGS[0]] if all(
            score_value(scores[s][cid], metric) is not None
            and isinstance(predictions[s][cid].get("generation_latency_seconds"), (int, float))
            and predictions[s][cid].get("status") == "success" for s in DEFAULT_SETTINGS)]
        for setting, label in zip(DEFAULT_SETTINGS, labels):
            quality = np.mean([score_value(scores[setting][cid], metric) for cid in cohort])
            times = [predictions[setting][cid]["generation_latency_seconds"] for cid in cohort]
            tokens = int(setting.split("-")[1]) * int(setting.split("-")[-1])
            x = float(np.median(times))
            tradeoff_rows.append({"metric": metric, "setting": setting, "n_paired_quality_and_latency": len(cohort), "mean_quality": quality, "median_generation_seconds": x, "retrieved_tokens": tokens})
            ax.scatter(x, quality, s=tokens / 5, alpha=.7)
            offset = (-65, 8) if label == "512/k=3" else (4, 8)
            ax.annotate(label, (x, quality), xytext=offset, textcoords="offset points", fontsize=8)
        ax.set(xlabel="Mediana da geração (s)", ylabel="Pontuação média", title=f"{metric}\n{len(cohort)} perguntas pareadas")
        ax.margins(x=.15, y=.15)
        ax.grid(alpha=.25)
    fig.suptitle("Qualidade e latência — área proporcional aos tokens recuperados")
    fig.tight_layout()
    fig.savefig(figures / "quality_latency_tradeoff.png", dpi=180)
    plt.close(fig)
    pd.DataFrame(tradeoff_rows).to_csv(tables / "quality_latency_paired.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for setting, label in zip(DEFAULT_SETTINGS, labels):
        group = frame[frame.setting == setting].dropna(subset=["generation_seconds"])
        axes[0].scatter(group.response_characters + 1, group.generation_seconds, s=14, alpha=.55, label=label)
    axes[0].set(xscale="log", yscale="log", xlabel="Caracteres da resposta + 1", ylabel="Geração (s)", title="Comprimento e latência")
    axes[0].legend(fontsize=8)
    arrays = [frame.loc[frame.setting == s, "generation_seconds"].dropna() for s in DEFAULT_SETTINGS]
    axes[1].boxplot(arrays)
    axes[1].set(yscale="log", ylabel="Geração (s)", title="Distribuição com todas as latências registradas")
    axes[1].set_xticks(range(1, 5), labels)
    for ax in axes:
        ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(figures / "latency_response_length.png", dpi=180)
    plt.close(fig)

    # Verify that portable exported tables exactly preserve the audit statistics.
    for original, name in exports.items():
        assert (analysis / original).read_bytes() == (tables / name).read_bytes()
    for name, digest in audit["input_sha256"].items():
        assert hashlib.sha256((source / name).read_bytes()).hexdigest() == digest
    metadata = {
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_name": source.name,
        "raw_input_sha256": audit["input_sha256"],
        "software": {p: version(p) for p in ("numpy", "pandas", "scipy", "matplotlib")},
        "n_unique_questions": audit["n_case_ids"],
        "primary_analysis_excludes_long_responses": False,
        "repetition_indicator": "1 - distinct 4grams / total 4grams in final 2000 whitespace tokens; diagnostic, not a validity label",
        "threshold_sensitivity": "5000, 10000 and 20000 characters, descriptive only",
        "exports_sha256": {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
                           for directory in (tables, figures) for p in sorted(directory.iterdir()) if p.is_file()},
    }
    (out / "provenance.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared delivery at {out}")
    print(pd.DataFrame(diagnostic_rows).to_string(index=False))
    print(correlations.to_string(index=False))


if __name__ == "__main__":
    main()
