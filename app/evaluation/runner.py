"""Offline generation: python -m app.evaluation.runner (from repository root)."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from app.backend.pipeline import run_query
from app.backend.prompts import CHAT_TEMPLATE


REPO_ROOT = Path(__file__).resolve().parents[2]


def run_evaluation(
    cases_path: Path,
    output_dir: Path,
    llm,
    prompt_template: str = CHAT_TEMPLATE,
) -> Path:
    """Generate predictions, retaining a record for every nonblank input line."""
    timestamp = datetime.now(timezone.utc)
    run_id = f"{timestamp:%Y%m%dT%H%M%SZ}_{uuid4().hex[:12]}"
    manifest = {
        "run_id": run_id,
        "model": llm.model,
        "temperature": llm.temperature,
        "prompt_template": prompt_template,
        "timestamp": timestamp.isoformat(),
        "cases_path": str(cases_path.resolve()),
        "total_cases": 0,
        "successful_cases": 0,
        "error_cases": 0,
    }

    # Open input before creating outputs so a missing dataset leaves no empty run.
    with cases_path.open(encoding="utf-8") as cases_file:
        run_dir = output_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        with (run_dir / "predictions.jsonl").open("w", encoding="utf-8") as predictions:
            for line_number, line in enumerate(cases_file, start=1):
                if not line.strip():
                    continue
                manifest["total_cases"] += 1
                record = {
                    "run_id": run_id,
                    "case_id": f"line_{line_number}",
                    "line_number": line_number,
                    "user_input": None,
                    "response": None,
                    "retrieved_contexts": [],
                }
                started = perf_counter()
                try:
                    case = json.loads(line)
                    if not isinstance(case, dict):
                        raise ValueError("Each case must be a JSON object")
                    record["case_id"] = case.get("case_id", record["case_id"])
                    if "reference" in case:
                        record["reference"] = case["reference"]
                    question = case.get("user_input")
                    if not isinstance(question, str) or not question.strip():
                        raise ValueError("user_input must be a nonempty string")
                    record["user_input"] = question
                    result = run_query(question, llm, prompt_template)
                    record.update(asdict(result))
                    record["status"] = "success"
                    manifest["successful_cases"] += 1
                except Exception as exc:
                    record.update(
                        status="error",
                        error_type=type(exc).__name__,
                        error=str(exc),
                        latency_seconds=perf_counter() - started,
                    )
                    manifest["error_cases"] += 1
                predictions.write(json.dumps(record, ensure_ascii=False) + "\n")
                predictions.flush()

    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate offline QA predictions with Ollama")
    parser.add_argument(
        "--cases", type=Path, default=REPO_ROOT / "evaluation_data" / "cases.jsonl"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=REPO_ROOT / "evaluation_results"
    )
    args = parser.parse_args()

    from app.backend.llm import get_llm

    run_dir = run_evaluation(args.cases, args.output_dir, get_llm())
    print(f"Results saved to {run_dir}")


if __name__ == "__main__":
    main()
