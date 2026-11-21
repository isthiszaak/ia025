"""Supplement the existing analysis without modifying source experiment files.

Run from the repository root:
  python scripts/analysis/audit_rag_configurations.py
Outputs are exploratory; thresholds and multiplicity families are explicit.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from app.evaluation.analyze_end_to_end import (
    DEFAULT_SETTINGS, METRIC_LABELS, holm_adjust, load_experiment,
    score_value, validate_and_index,
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--experiment-dir", type=Path, default=ROOT / "files_ia_025/consolidated_120_cases_20261108T165307Z")
parser.add_argument("--analysis-dir", type=Path, default=ROOT / "evaluation_results/rag_configuration_analysis/reproduced")
parser.add_argument("--output-dir", type=Path, default=ROOT / "evaluation_results/rag_configuration_analysis/audit")
args = parser.parse_args()
SOURCE = args.experiment_dir.resolve()
OUT = args.output_dir.resolve()
OUT.mkdir(parents=True, exist_ok=True)
SETTINGS = list(DEFAULT_SETTINGS)
PERMUTATIONS = 25000
LONG_RESPONSE_CHARACTERS = 10000  # Diagnostic threshold, never an exclusion.
rng = np.random.default_rng(42)
manifest, comparison, approaches, metrics = load_experiment(SOURCE)
case_ids, indexed = validate_and_index(approaches)

input_hashes = {
    str(path.relative_to(SOURCE)): hashlib.sha256(path.read_bytes()).hexdigest()
    for path in SOURCE.rglob("*")
    if path.is_file() and path.suffix in {".json", ".jsonl"}
    and not {"analysis", "notebook_analysis"}.intersection(path.relative_to(SOURCE).parts)
}
question_reference_consistent = all(
    len({(indexed[s]["predictions"][cid]["user_input"],
          indexed[s]["predictions"][cid].get("reference")) for s in SETTINGS}) == 1
    for cid in case_ids
)
assert question_reference_consistent

friedman_rows, paired_means, prediction_rows = [], [], []
for metric in metrics:
    matrix = np.array([
        [score_value(indexed[s]["scores"][cid], metric) for s in SETTINGS]
        for cid in case_ids
        if all(score_value(indexed[s]["scores"][cid], metric) is not None for s in SETTINGS)
    ], dtype=float)
    n, k = matrix.shape
    ranks = stats.rankdata(matrix, axis=1)
    observed = float(np.sum((ranks.sum(axis=0) - n * (k + 1) / 2) ** 2))
    if observed == 0 and np.all(np.ptp(matrix, axis=1) == 0):
        q, asymptotic_p, permutation_p = 0.0, 1.0, 1.0
    else:
        result = stats.friedmanchisquare(*matrix.T)
        q, asymptotic_p = float(result.statistic), float(result.pvalue)
        extreme, completed = 0, 0
        while completed < PERMUTATIONS:
            batch = min(500, PERMUTATIONS - completed)
            permuted = rng.permuted(np.broadcast_to(ranks, (batch, n, k)), axis=2)
            null_statistics = ((permuted.sum(axis=1) - n * (k + 1) / 2) ** 2).sum(axis=1)
            extreme += int(np.count_nonzero(null_statistics >= observed - 1e-10))
            completed += batch
        permutation_p = (extreme + 1) / (PERMUTATIONS + 1)
    friedman_rows.append({
        "metric": metric, "n_complete": n, "friedman_q": q,
        "kendalls_w": q / (n * (k - 1)), "p_asymptotic": asymptotic_p,
        "p_permutation": permutation_p, "permutations": PERMUTATIONS,
        "permutation_mc_standard_error": np.sqrt(permutation_p * (1 - permutation_p) / (PERMUTATIONS + 1)),
    })
    for setting, mean in zip(SETTINGS, matrix.mean(axis=0)):
        paired_means.append({"metric": metric, "setting": setting,
                             "n_complete": n, "mean_same_complete_cohort": mean})
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    for chunk, columns in ((256, (0, 1)), (512, (2, 3))):
        ax.plot([1, 3], matrix[:, columns].mean(axis=0), marker="o", label=f"Chunk {chunk}")
    ax.set(xticks=[1, 3], xlabel="Top-k", ylabel="Pontuação média",
           title=f"{METRIC_LABELS.get(metric, metric)} — {n} perguntas completas")
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / f"interaction_{metric}.png", dpi=180)
    plt.close(fig)

friedman = pd.DataFrame(friedman_rows)
friedman["p_permutation_holm_7_metrics"] = holm_adjust(friedman.p_permutation.tolist())
friedman.to_csv(OUT / "friedman_permutation.csv", index=False)
pd.DataFrame(paired_means).to_csv(OUT / "quality_same_complete_cohort.csv", index=False)

latency_rows = []
for setting in SETTINGS:
    data = indexed[setting]
    for cid in case_ids:
        pred, retrieval = data["predictions"][cid], data["retrieval"][cid]
        assert pred["retrieved_contexts"] == retrieval["retrieved_contexts"]
        assert pred["user_input"] == retrieval["user_input"]
        chunks = retrieval["retrieved_chunks"]
        counts = [c.get("metadata", {}).get("token_count") for c in chunks]
        tokens = sum(counts) if all(isinstance(c, (int, float)) for c in counts) else None
        response_chars = len(pred.get("response") or "")
        row = {
            "case_id": cid, "setting": setting,
            "generation_status": pred.get("generation_status"),
            "error": pred.get("error"), "response_chars": response_chars,
            "long_response_diagnostic_only": response_chars > LONG_RESPONSE_CHARACTERS,
            "chunks": len(chunks), "retrieved_tokens_embedding_tokenizer": tokens,
            "retrieval_seconds": retrieval.get("latency_seconds"),
            "generation_seconds": pred.get("generation_latency_seconds"),
        }
        prediction_rows.append(row)
    group = pd.DataFrame([r for r in prediction_rows if r["setting"] == setting])
    valid = group.dropna(subset=["generation_seconds"])
    long = valid[valid.long_response_diagnostic_only]
    latency_rows.append({
        "setting": setting, "n_cases": len(group), "n_generation_timed": len(valid),
        "n_generation_errors": int((group.generation_status != "success").sum()),
        "generation_mean_seconds_all": valid.generation_seconds.mean(),
        "generation_median_seconds_all": valid.generation_seconds.median(),
        "generation_p95_seconds_all": valid.generation_seconds.quantile(.95),
        "generation_total_seconds": valid.generation_seconds.sum(),
        "n_responses_gt10000_chars": len(long),
        "generation_time_fraction_responses_gt10000_chars": long.generation_seconds.sum() / valid.generation_seconds.sum(),
        "max_response_chars": int(group.response_chars.max()),
        "retrieval_median_seconds": group.retrieval_seconds.median(),
        "retrieved_tokens_mean": group.retrieved_tokens_embedding_tokenizer.mean(),
    })
predictions = pd.DataFrame(prediction_rows)
predictions.to_csv(OUT / "prediction_latency_audit.csv", index=False)
latency = pd.DataFrame(latency_rows)
latency.to_csv(OUT / "latency_tail_diagnostics.csv", index=False)

reproduced = args.analysis_dir.resolve()
correlations = pd.read_csv(reproduced / "retrieval_correlations.csv")
correlations["p_value_holm_all_testable_correlations"] = holm_adjust(correlations.p_value.tolist())
correlations["interpretation"] = np.where(
    correlations.spearman_rho.isna(), "undefined_or_insufficient_variation", "exploratory_approximate_p_value")
correlations.to_csv(OUT / "retrieval_correlations_adjusted.csv", index=False)
posthoc = pd.read_csv(reproduced / "rag_configuration_posthoc.csv")
posthoc["p_holm_all_42_exploratory_comparisons"] = holm_adjust(posthoc.p_value.tolist())
posthoc.to_csv(OUT / "posthoc_multiplicity_sensitivity.csv", index=False)
factorial = pd.read_csv(reproduced / "factorial_effects.csv")
factorial["p_holm_all_21_exploratory_contrasts"] = holm_adjust(factorial.p_value.tolist())
factorial.to_csv(OUT / "factorial_multiplicity_sensitivity.csv", index=False)

# Descriptive trade-off only: varying complete cohorts are explicitly labelled.
means = pd.DataFrame(paired_means)
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
for ax, metric in zip(axes, ("answer_relevancy", "faithfulness", "context_recall")):
    for setting in SETTINGS:
        row = latency[latency.setting == setting].iloc[0]
        quality = means[(means.setting == setting) & (means.metric == metric)].iloc[0]
        label = setting.replace("tokens-", "").replace("-topk-", "/k=")
        ax.scatter(row.generation_median_seconds_all, quality.mean_same_complete_cohort,
                   s=row.retrieved_tokens_mean / 5, alpha=.7)
        ax.annotate(label, (row.generation_median_seconds_all, quality.mean_same_complete_cohort),
                    xytext=(3, 6), textcoords="offset points", fontsize=8)
    ax.set(xlabel="Mediana da geração (s; todos os tempos válidos)",
           ylabel="Qualidade média (coorte completa por métrica)", title=METRIC_LABELS[metric])
    ax.grid(alpha=.25)
fig.suptitle("Qualidade × latência — área dos pontos proporcional aos tokens recuperados")
fig.tight_layout()
fig.savefig(OUT / "quality_latency_tradeoff.png", dpi=180)
plt.close(fig)

sources = sorted({chunk["source"] for setting in SETTINGS
                  for r in indexed[setting]["retrieval"].values() for chunk in r["retrieved_chunks"]})
audit = {
    "source": str(SOURCE), "n_case_ids": len(case_ids),
    "same_questions_and_references": question_reference_consistent,
    "prediction_contexts_match_retrieval": True,
    "retrieved_source_files": sources,
    "corpus_selection": manifest["corpus"]["selection"],
    "source_runs": manifest.get("source_runs"),
    "input_sha256": input_hashes,
    "seed": 42, "friedman_permutations": PERMUTATIONS,
    "long_response_threshold_characters": LONG_RESPONSE_CHARACTERS,
    "long_responses_excluded_from_primary_analysis": False,
    "limitations": [
        "Corpus restricted to three selected norms; no generalization to all 461 documents.",
        "Retrieved token counts are constant within each setting; within-setting token/quality correlation is undefined.",
        "Token counts use the embedding tokenizer, not actual generation prompt/output token usage.",
        "Long and repetitive outputs strongly affect mean latency; retained in primary analysis.",
        "No hardware/energy/VRAM or generator token-usage logs available in these records.",
        "Four source batches: hardware and generation sampling settings are not fully established by consolidated metadata.",
        "Wilcoxon tests paired-difference ranks; pointwise bootstrap mean intervals target a different estimand.",
        "Multiplicity families are reported separately; local-family significance is not global-family significance.",
        "Spearman p-values are approximate and exploratory; within-setting analysis is not a causal test.",
        "Trade-off figure combines complete-case quality means and all timed generation records; descriptive only.",
    ],
}
(OUT / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
assert all(hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() == digest
           for name, digest in input_hashes.items())
print(f"Audit complete: {OUT}")
print(friedman[["metric", "n_complete", "kendalls_w", "p_permutation_holm_7_metrics"]].to_string(index=False))
