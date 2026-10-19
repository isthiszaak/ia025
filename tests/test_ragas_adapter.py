import asyncio
from dataclasses import asdict
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from app.evaluation.ragas_adapter import (
    EvaluatorConfig, build_metrics, evaluate_predictions, score_case,
)


class FakeMetric:
    def __init__(self, value=0.5):
        self.value = value
        self.calls = []

    async def ascore(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs["response"] == "fail":
            raise RuntimeError("simulated failure")
        return SimpleNamespace(value=self.value)


class AdapterTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.config = EvaluatorConfig("judge-model", "embedding-model")
        self.metrics = {"answer_relevancy": FakeMetric(0.5),
                        "answer_correctness": FakeMetric(0.75)}

    async def test_missing_reference_and_disabled_retrieval_are_null(self):
        record = await score_case(
            {"user_input": "question", "response": "answer", "retrieved_contexts": []},
            self.metrics, 1,
        )
        self.assertEqual(record["scores"]["answer_relevancy"], 0.5)
        for name in ("answer_correctness", "faithfulness", "context_precision", "context_recall"):
            self.assertIsNone(record["scores"][name])
            self.assertEqual(record["metric_status"][name], "not_applicable")
        self.assertEqual(self.metrics["answer_correctness"].calls, [])

    async def test_metric_failure_does_not_prevent_other_metric(self):
        async def fail(**kwargs):
            raise RuntimeError("judge failed")

        self.metrics["answer_relevancy"].ascore = fail
        record = await score_case(
            {"user_input": "question", "response": "answer", "reference": "expected"},
            self.metrics, 1,
        )
        self.assertIsNone(record["scores"]["answer_relevancy"])
        self.assertEqual(record["metric_status"]["answer_relevancy"], "error")
        self.assertEqual(record["scores"]["answer_correctness"], 0.75)
        self.assertEqual(self.metrics["answer_correctness"].calls[0]["reference"], "expected")

    async def test_nonfinite_and_timeout_are_failures_but_zero_is_valid(self):
        case = {"user_input": "q", "response": "a"}
        self.metrics["answer_relevancy"].value = float("nan")
        record = await score_case(case, self.metrics, 1)
        self.assertEqual(record["metric_status"]["answer_relevancy"], "error")
        self.assertIsNone(record["scores"]["answer_relevancy"])

        async def slow(**kwargs):
            await asyncio.sleep(1)

        self.metrics["answer_relevancy"].ascore = slow
        record = await score_case(case, self.metrics, 0.001)
        self.assertEqual(record["metric_status"]["answer_relevancy"], "error")
        self.metrics["answer_relevancy"] = FakeMetric(0)
        record = await score_case(case, self.metrics, 1)
        self.assertEqual(record["metric_status"]["answer_relevancy"], "success")
        self.assertEqual(record["scores"]["answer_relevancy"], 0)

    async def test_saved_predictions_errors_summary_and_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            predictions = root / "predictions.jsonl"
            cases = [
                {"case_id": "a", "user_input": "q", "response": "a", "reference": "r"},
                {"case_id": "b", "user_input": "q", "response": "fail"},
                {"case_id": "c", "user_input": "q", "response": None, "status": "error"},
                {"case_id": "d", "user_input": "q", "response": "last"},
            ]
            predictions.write_text(
                '\n'.join(json.dumps(case) for case in cases) + '\n{invalid\n[]\n\n',
                encoding="utf-8",
            )
            original = predictions.read_bytes()
            (root / "manifest.json").write_text(json.dumps({"run_id": "source", "model": "generator"}))
            await evaluate_predictions(predictions, root, self.metrics, self.config, "0.4.3")
            self.assertEqual(predictions.read_bytes(), original)
            rows = [json.loads(line) for line in (root / "scores.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 6)
            self.assertEqual(rows[2]["metric_status"]["answer_relevancy"], "skipped")
            self.assertEqual(rows[3]["scores"]["answer_relevancy"], 0.5)
            summary = json.loads((root / "summary.json").read_text())
            self.assertEqual(summary["total_cases"], 6)
            relevance = summary["metrics"]["answer_relevancy"]
            self.assertEqual(relevance, {"mean": 0.5, "stddev": 0.0, "valid_count": 2,
                                         "failure_count": 3, "not_applicable_count": 0, "skipped_count": 1})
            self.assertEqual(summary["metrics"]["answer_correctness"]["valid_count"], 1)
            self.assertIsNone(summary["metrics"]["faithfulness"]["mean"])
            self.assertIsNone(summary["metrics"]["faithfulness"]["stddev"])
            manifest = json.loads((root / "manifest.json").read_text())
            self.assertEqual(manifest["model"], "generator")
            self.assertEqual(manifest["evaluation"]["config"], asdict(self.config))
            self.assertEqual(manifest["evaluation"]["ragas_version"], "0.4.3")
            with self.assertRaises(FileExistsError):
                await evaluate_predictions(predictions, root, self.metrics, self.config, "0.4.3")

    async def test_empty_input_has_null_statistics(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            predictions = root / "predictions.jsonl"
            predictions.write_text("\n")
            await evaluate_predictions(predictions, root / "eval", self.metrics, self.config, "test")
            summary = json.loads((root / "eval" / "summary.json").read_text())
            self.assertEqual(summary["total_cases"], 0)
            for metric in summary["metrics"].values():
                self.assertIsNone(metric["mean"])
                self.assertIsNone(metric["stddev"])
                self.assertEqual(metric["failure_count"], 0)

    async def test_population_standard_deviation(self):
        class SequenceMetric:
            async def ascore(self, **kwargs):
                return SimpleNamespace(value=float(kwargs["response"]))

        self.metrics["answer_relevancy"] = SequenceMetric()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            predictions = root / "predictions.jsonl"
            predictions.write_text('\n'.join(json.dumps({"user_input": "q", "response": str(v)}) for v in (0, 1)))
            await evaluate_predictions(predictions, root, self.metrics, self.config, "test")
            metric = json.loads((root / "summary.json").read_text())["metrics"]["answer_relevancy"]
            self.assertEqual((metric["mean"], metric["stddev"]), (0.5, 0.5))

    def test_config_never_inherits_generator_model(self):
        with patch.dict(os.environ, {"OLLAMA_MODEL": "generator"}, clear=True):
            with self.assertRaises(ValueError):
                EvaluatorConfig.from_env()

    @unittest.skipUnless(importlib.util.find_spec("ragas"), "optional RAGAS dependency not installed")
    async def test_real_ragas_collections_and_factories_with_mock_http(self):
        import httpx
        from openai import AsyncOpenAI

        requests = []

        def handler(request):
            body = json.loads(request.content)
            requests.append((request.url.path, body))
            if request.url.path.endswith("/embeddings"):
                texts = body["input"]
                texts = [texts] if isinstance(texts, str) else texts
                data = [{"object": "embedding", "index": i, "embedding": [1.0, 0.0]} for i in range(len(texts))]
                return httpx.Response(200, json={"object": "list", "data": data, "model": body["model"],
                                                "usage": {"prompt_tokens": 1, "total_tokens": 1}})
            content = '\n'.join(str(message.get("content", "")) for message in body["messages"])
            if "noncommittal" in content:
                output = {"question": "question", "noncommittal": 0}
            elif '"TP"' in content:
                output = {"TP": [{"statement": "answer", "reason": "supported"}], "FP": [], "FN": []}
            else:
                output = {"statements": ["answer"]}
            return httpx.Response(200, json={
                "id": "mock", "object": "chat.completion", "created": 0, "model": body["model"],
                "choices": [{"index": 0, "message": {"role": "assistant", "content": json.dumps(output)},
                             "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            })

        class MockAsyncOpenAI(AsyncOpenAI):
            def __init__(self, **kwargs):
                super().__init__(
                    **kwargs, http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
                )

        with patch("openai.AsyncOpenAI", MockAsyncOpenAI):
            metrics, clients = build_metrics(self.config)
        try:
            record = await score_case(
                {"user_input": "question", "response": "answer", "reference": "answer"}, metrics, 10,
            )
            self.assertEqual(record["metric_status"]["answer_relevancy"], "success", record)
            self.assertEqual(record["metric_status"]["answer_correctness"], "success", record)
            self.assertAlmostEqual(record["scores"]["answer_relevancy"], 1.0)
            self.assertAlmostEqual(record["scores"]["answer_correctness"], 1.0)
            self.assertTrue(any(body["model"] == "embedding-model" for _, body in requests))
            self.assertTrue(any(body["model"] == "judge-model" for _, body in requests))
        finally:
            for client in clients:
                await client.close()


if __name__ == "__main__":
    unittest.main()
