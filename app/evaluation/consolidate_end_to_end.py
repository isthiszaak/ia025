#!/usr/bin/env python3
"""
Consolida automaticamente execuções completas do pipeline RAG vs. LLM-only.

Uso (a partir da raiz do projeto):
    python -m app.evaluation.consolidate_end_to_end

Ou:
    python app/evaluation/consolidate_end_to_end.py \
        --root app/evaluation/evaluation_results/end_to_end

O script:
- descobre execuções originais diretamente dentro de --root;
- valida compatibilidade de modelos, métricas e corpus;
- valida os mesmos case_ids nas quatro configurações e no LLM-only;
- impede duplicação de case_id entre execuções;
- concatena predictions/retrieval/pairs/scores;
- corrige prediction_line_number nos scores consolidados;
- recalcula médias e desvios-padrão populacionais a partir dos scores individuais;
- cria uma nova pasta, sem modificar as execuções originais.

Por segurança, qualquer incompatibilidade ou duplicidade interrompe a consolidação.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


EXPECTED_SETTINGS = (
    "tokens-256-topk-1",
    "tokens-256-topk-3",
    "tokens-512-topk-1",
    "tokens-512-topk-3",
)
APPROACHES = (*EXPECTED_SETTINGS, "llm_only")
REQUIRED_RUN_FILES = ("manifest.json", "comparison.json")
JSONL_FILES = {
    "predictions": "predictions.jsonl",
    "retrieval": "retrieval.jsonl",
    "pairs": "pairs.jsonl",
    "scores": "evaluation/scores.jsonl",
}
SETTING_PATTERN = re.compile(r"^tokens-(\d+)-topk-(\d+)$")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Esperado objeto JSON em: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"JSONL inválido em {path}, linha {line_number}: {exc}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"Registro não é objeto JSON em {path}, linha {line_number}"
                )
            records.append(record)
    return records


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=False, indent=2, allow_nan=False)
        file.write("\n")


def write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False, allow_nan=False))
            file.write("\n")


def discover_runs(root: Path) -> list[Path]:
    runs = []
    for candidate in sorted(root.iterdir()):
        if not candidate.is_dir():
            continue
        if not all((candidate / name).is_file() for name in REQUIRED_RUN_FILES):
            continue
        manifest = read_json(candidate / "manifest.json")
        if manifest.get("stage") != "end_to_end_rag_vs_llm_only":
            continue
        if manifest.get("state") != "complete":
            print(
                f"[AVISO] Ignorando execução incompleta: {candidate.name}",
                file=sys.stderr,
            )
            continue
        runs.append(candidate)
    return runs


def compatibility_signature(run_dir: Path) -> dict[str, Any]:
    manifest = read_json(run_dir / "manifest.json")
    comparison = read_json(run_dir / "comparison.json")

    signature: dict[str, Any] = {
        "execution_mode": manifest.get("execution_mode"),
        "protocol": manifest.get("protocol"),
        "chunk_overlap_tokens": manifest.get("chunk_overlap_tokens"),
        "generation_model": manifest.get("generation_model"),
        "ragas_version": manifest.get("ragas_version"),
        "enabled_metrics": sorted(manifest.get("enabled_metrics") or comparison.get("metrics") or []),
        "corpus_sha256": (manifest.get("corpus") or {}).get("corpus_sha256"),
        "selection_sha256": (manifest.get("corpus") or {}).get("selection_sha256"),
    }

    # Configuração de avaliação: modelos, endpoints e parâmetros do avaliador.
    evaluation_manifest_path = run_dir / "tokens-256-topk-1" / "evaluation" / "manifest.json"
    if not evaluation_manifest_path.exists():
        raise ValueError(f"Manifest de avaliação ausente em {run_dir}")
    evaluation_manifest = read_json(evaluation_manifest_path)
    evaluation_config = (evaluation_manifest.get("evaluation") or {}).get("config") or {}
    signature["evaluation_config"] = {
        key: value
        for key, value in evaluation_config.items()
        if key not in {"selected_metrics"}
    }

    # Configuração do embedding/retrieval, ignorando caminhos locais e IDs de coleção.
    retrieval_manifest_path = run_dir / "tokens-256-topk-1" / "manifest.json"
    if retrieval_manifest_path.exists():
        retrieval_manifest = read_json(retrieval_manifest_path)
        retrieval_config = retrieval_manifest.get("config") or {}
        signature["retrieval_config"] = {
            key: value
            for key, value in retrieval_config.items()
            if key not in {
                "data_dir", "index_dir", "corpus_selection", "base_url",
            }
        }
    else:
        signature["retrieval_config"] = None

    return signature


def validate_run(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any], set[str]]:
    manifest = read_json(run_dir / "manifest.json")
    comparison = read_json(run_dir / "comparison.json")

    if comparison.get("run_id") != manifest.get("run_id"):
        raise ValueError(f"run_id divergente em {run_dir}")

    settings = {item.get("setting") for item in comparison.get("configurations", [])}
    if settings != set(EXPECTED_SETTINGS):
        raise ValueError(
            f"{run_dir.name}: configurações encontradas {sorted(settings)}; "
            f"esperadas {list(EXPECTED_SETTINGS)}"
        )

    ids_by_approach: dict[str, set[str]] = {}
    for approach in APPROACHES:
        prediction_path = run_dir / approach / "predictions.jsonl"
        score_path = run_dir / approach / "evaluation" / "scores.jsonl"
        if not prediction_path.exists() or not score_path.exists():
            raise ValueError(
                f"{run_dir.name}: faltam predictions ou scores para {approach}"
            )
        predictions = read_jsonl(prediction_path)
        scores = read_jsonl(score_path)
        prediction_ids = [str(item.get("case_id", "")) for item in predictions]
        score_ids = [str(item.get("case_id", "")) for item in scores]
        if not prediction_ids or any(not case_id for case_id in prediction_ids):
            raise ValueError(f"{run_dir.name}/{approach}: predictions sem case_id")
        if len(prediction_ids) != len(set(prediction_ids)):
            raise ValueError(f"{run_dir.name}/{approach}: case_id duplicado em predictions")
        if set(prediction_ids) != set(score_ids):
            raise ValueError(
                f"{run_dir.name}/{approach}: case_ids de predictions e scores diferem"
            )
        ids_by_approach[approach] = set(prediction_ids)

    reference_ids = ids_by_approach["llm_only"]
    for approach in EXPECTED_SETTINGS:
        if ids_by_approach[approach] != reference_ids:
            raise ValueError(
                f"{run_dir.name}: case_ids de {approach} não correspondem ao LLM-only"
            )

    declared_count = manifest.get("cases_count")
    if declared_count is not None and int(declared_count) != len(reference_ids):
        raise ValueError(
            f"{run_dir.name}: manifest declara {declared_count} casos, "
            f"mas foram encontrados {len(reference_ids)}"
        )
    return manifest, comparison, reference_ids


def metric_summary(score_records: list[dict[str, Any]], metrics: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for metric in metrics:
        values: list[float] = []
        durations: list[float] = []
        attempted = failures = not_applicable = skipped = 0

        for record in score_records:
            status = (record.get("metric_status") or {}).get(metric)
            score = (record.get("scores") or {}).get(metric)
            duration = (record.get("metric_duration_seconds") or {}).get(metric)

            if status == "not_applicable":
                not_applicable += 1
            elif status == "skipped":
                skipped += 1
            elif status in {"success", "error", "failed", "failure"}:
                attempted += 1
                if status == "success" and isinstance(score, (int, float)) and math.isfinite(float(score)):
                    values.append(float(score))
                elif status != "success":
                    failures += 1
            elif status is not None:
                # Status desconhecido não é convertido silenciosamente em sucesso.
                attempted += 1
                failures += 1

            if isinstance(duration, (int, float)) and math.isfinite(float(duration)):
                durations.append(float(duration))

        total_duration = sum(durations)
        result[metric] = {
            "mean": statistics.fmean(values) if values else None,
            "stddev": statistics.pstdev(values) if len(values) > 1 else (0.0 if values else None),
            "total_duration_seconds": round(total_duration, 6),
            "mean_duration_seconds": (
                round(total_duration / attempted, 6) if attempted else None
            ),
            "attempted_count": attempted,
            "valid_count": len(values),
            "failure_count": failures,
            "not_applicable_count": not_applicable,
            "skipped_count": skipped,
        }
    return result


def merge_jsonl_for_approach(
    run_dirs: list[Path],
    output_dir: Path,
    approach: str,
) -> tuple[dict[str, list[dict[str, Any]]], int]:
    merged: dict[str, list[dict[str, Any]]] = {
        "predictions": [],
        "retrieval": [],
        "pairs": [],
        "scores": [],
    }
    prediction_offset = 0

    for run_dir in run_dirs:
        approach_dir = run_dir / approach
        for key, relative_path in JSONL_FILES.items():
            path = approach_dir / relative_path
            records = read_jsonl(path)
            if key == "scores":
                for record in records:
                    if isinstance(record.get("prediction_line_number"), int):
                        record["prediction_line_number"] += prediction_offset
            merged[key].extend(records)

        prediction_offset += len(read_jsonl(approach_dir / "predictions.jsonl"))

    for key, relative_path in JSONL_FILES.items():
        write_jsonl(output_dir / approach / relative_path, merged[key])

    return merged, len(merged["predictions"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("app/evaluation/evaluation_results/end_to_end"),
        help="Diretório que contém as execuções originais.",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Nome da pasta consolidada (padrão: consolidated_<timestamp>).",
    )
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Diretório não encontrado: {root}")

    run_dirs = discover_runs(root)
    if not run_dirs:
        raise SystemExit(f"Nenhuma execução completa encontrada em {root}")

    validated = []
    for run_dir in run_dirs:
        manifest, comparison, case_ids = validate_run(run_dir)
        validated.append((run_dir, manifest, comparison, case_ids))

    # Todas as execuções precisam compartilhar a mesma configuração experimental.
    signatures = [(run_dir.name, compatibility_signature(run_dir)) for run_dir in run_dirs]
    baseline_name, baseline_signature = signatures[0]
    incompatible = [
        (name, signature)
        for name, signature in signatures[1:]
        if signature != baseline_signature
    ]
    if incompatible:
        details = [baseline_name, *[name for name, _ in incompatible]]
        raise SystemExit(
            "As execuções não possuem configurações compatíveis. "
            f"Verifique modelos, métricas, corpus e parâmetros: {details}"
        )

    # Detecta sobreposição de perguntas entre batches antes de gravar qualquer coisa.
    seen_ids: dict[str, str] = {}
    for run_dir, _, _, case_ids in validated:
        for case_id in case_ids:
            if case_id in seen_ids:
                raise SystemExit(
                    f"case_id repetido entre execuções: {case_id!r} aparece em "
                    f"{seen_ids[case_id]} e {run_dir.name}. "
                    "A consolidação foi cancelada para evitar duplicações."
                )
            seen_ids[case_id] = run_dir.name

    total_cases = len(seen_ids)
    metrics = list(validated[0][2].get("metrics") or [])
    if not metrics:
        raise SystemExit("Nenhuma métrica declarada em comparison.json")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_name = args.name or f"consolidated_{total_cases}_cases_{timestamp}"
    output_dir = root / output_name
    if output_dir.exists():
        raise SystemExit(f"A pasta de destino já existe: {output_dir}")
    output_dir.mkdir(parents=True)

    merged_by_approach: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for approach in APPROACHES:
        merged, count = merge_jsonl_for_approach(run_dirs, output_dir, approach)
        if count != total_cases:
            raise SystemExit(
                f"{approach}: foram consolidadas {count} predictions, "
                f"mas eram esperadas {total_cases}"
            )
        merged_by_approach[approach] = merged

        summary = {
            "total_cases": count,
            "standard_deviation": "population",
            "selected_metrics": metrics,
            "metrics": metric_summary(merged["scores"], metrics),
        }
        write_json(output_dir / approach / "evaluation" / "summary.json", summary)

        # Manifesto de avaliação baseado no primeiro batch, com caminhos consolidados.
        source_eval_manifest = read_json(
            run_dirs[0] / approach / "evaluation" / "manifest.json"
        )
        evaluation = source_eval_manifest.get("evaluation", {})
        evaluation["total_cases"] = count
        evaluation["predictions_path"] = str(
            output_dir / approach / "predictions.jsonl"
        )
        evaluation["output_dir"] = str(output_dir / approach / "evaluation")
        evaluation["persistence"] = "consolidated_from_completed_runs"
        write_json(
            output_dir / approach / "evaluation" / "manifest.json",
            {"evaluation": evaluation},
        )

    configurations = []
    for setting in EXPECTED_SETTINGS:
        match = SETTING_PATTERN.match(setting)
        assert match is not None
        chunk_size, top_k = map(int, match.groups())
        configurations.append({
            "setting": setting,
            "chunk_size": chunk_size,
            "top_k": top_k,
            "metrics": metric_summary(
                merged_by_approach[setting]["scores"], metrics
            ),
            "total_cases": total_cases,
        })

    comparison = {
        "run_id": output_name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "consolidated_end_to_end_rag_vs_llm_only",
        "metrics": metrics,
        "source_runs": [run_dir.name for run_dir in run_dirs],
        "total_cases": total_cases,
        "configurations": configurations,
        "llm_only": {
            "setting": "llm_only",
            "metrics": metric_summary(
                merged_by_approach["llm_only"]["scores"], metrics
            ),
            "total_cases": total_cases,
        },
    }
    write_json(output_dir / "comparison.json", comparison)

    # Manifesto raiz: preserva a configuração comum e documenta a origem.
    consolidated_manifest = dict(validated[0][1])
    consolidated_manifest.update({
        "run_id": output_name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "consolidated_end_to_end_rag_vs_llm_only",
        "state": "complete",
        "execution_mode": "consolidated_all_combinations",
        "cases_count": total_cases,
        "comparisons_count": len(EXPECTED_SETTINGS),
        "retrieval_count": sum(
            len(merged_by_approach[s]["retrieval"]) for s in EXPECTED_SETTINGS
        ),
        "errors": 0,
        "output_dir": str(output_dir),
        "source_runs": [
            {
                "run_id": manifest.get("run_id"),
                "directory": run_dir.name,
                "cases_count": len(case_ids),
                "timestamp_utc": manifest.get("timestamp_utc"),
            }
            for run_dir, manifest, _, case_ids in validated
        ],
        "comparison_path": str(output_dir / "comparison.json"),
        "llm_only_predictions_path": str(
            output_dir / "llm_only" / "predictions.jsonl"
        ),
        "consolidation": {
            "method": "concatenate_individual_records_and_recalculate_metrics",
            "source_runs_count": len(run_dirs),
            "unique_case_ids": total_cases,
            "duplicate_case_ids": 0,
            "configuration_signature": baseline_signature,
        },
    })
    consolidated_manifest["runs"] = [
        {
            "setting": setting,
            "chunk_size": int(SETTING_PATTERN.match(setting).group(1)),
            "top_k": int(SETTING_PATTERN.match(setting).group(2)),
            "path": setting,
            "status": "success",
            "errors": 0,
        }
        for setting in EXPECTED_SETTINGS
    ]
    write_json(output_dir / "manifest.json", consolidated_manifest)

    print("\nConsolidação concluída.")
    print(f"Execuções de origem: {len(run_dirs)}")
    for run_dir in run_dirs:
        print(f"  - {run_dir.name}")
    print(f"Perguntas únicas: {total_cases}")
    print(f"Configurações RAG: {len(EXPECTED_SETTINGS)}")
    print(f"Destino: {output_dir}")


if __name__ == "__main__":
    main()
