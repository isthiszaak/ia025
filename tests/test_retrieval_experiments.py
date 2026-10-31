"""Comparison orchestration tests with mocked retrieval; no Ollama service."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from app.retrieval.config import RetrievalConfig
from app.retrieval.experiments import run_comparisons


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = RetrievalConfig(data_dir=self.root / "data", index_dir=self.root / "index",
                                      corpus_selection=None, chunk_unit="tokens",
                                      chunk_size=512, chunk_overlap=50)
        prepared = self.config.data_dir / "prepared"
        prepared.mkdir(parents=True)
        self.corpus = b'{"id":"doc1","text":"Original corpus"}\n'
        (prepared / "corpus.jsonl").write_bytes(self.corpus)
        (prepared / "manifest.json").write_text(
            json.dumps({"dataset": "test-fixture", "revision": "test-rev",
                        "corpus_sha256": hashlib.sha256(self.corpus).hexdigest()}), encoding="utf-8")
        self.cases = [{"case_id": "q1", "user_input": "Primeira pergunta?", "reference": "Gold only"},
                      {"case_id": "q2", "user_input": "Segunda pergunta?"}]
        self.opened = []
        self.queries = []
        self.open_failure = None
        self.query_failure = None
        outer = self

        class FakeRetriever:
            def __init__(self, config):
                outer.opened.append(config)
                if config.chunk_size == outer.open_failure:
                    raise RuntimeError("fixture: incompatible index")
                self.config = config
                self.collection_name = f"test-collection-{config.chunk_size}"
                self.manifest = {"model_digest": "simulated-model-digest", "collection": self.collection_name,
                                 "corpus_sha256": hashlib.sha256(outer.corpus).hexdigest(),
                                 "parameters": {"chunk_size": config.chunk_size, "chunk_unit": "tokens"}}

            def retrieve(self, question, top_k=None):
                outer.queries.append((self.config.chunk_size, question, top_k))
                if question == outer.query_failure:
                    raise RuntimeError("fixture: embedding request failed")
                return [{"chunk_id": f"doc1-{i}", "document_id": "doc1", "source": "original.txt",
                         "title": "Original", "text": "Trecho original com acentuação.",
                         "metadata": {"source_aliases": ["original.txt", "duplicate.txt"],
                                      "start_char": 0, "end_char": 34},
                         "distance": 0.125 + i / 10, "distance_metric": "cosine"}
                        for i in range(top_k)]

        self.retriever_patch = patch("app.retrieval.experiments.Retriever", FakeRetriever)
        self.retriever_patch.start()
        self.addCleanup(self.retriever_patch.stop)
        self.dependencies_patch = patch("app.retrieval.experiments.dependency_versions",
                                        return_value={"python": "fixture", "tokenizers": "fixture"})
        self.dependencies_patch.start()
        self.addCleanup(self.dependencies_patch.stop)

    def run_comparisons(self, **kwargs):
        return run_comparisons(self.config, self.cases, self.root / "results", **kwargs)

    def read_run(self, summary, setting):
        path = Path(summary["output_dir"]) / setting
        return (json.loads((path / "manifest.json").read_text(encoding="utf-8")),
                [json.loads(line) for line in (path / "retrieval.jsonl").read_text(
                    encoding="utf-8").splitlines()])

    def test_only_five_planned_settings_preserve_overlap_and_reuse_top_k_index(self):
        with patch("app.retrieval.experiments.build_index") as builder:
            summary = self.run_comparisons()
        builder.assert_not_called()
        self.assertEqual([(run["chunk_size"], run["top_k"]) for run in summary["runs"]],
                         [(256, 3), (512, 3), (1024, 3), (512, 1), (512, 5)])
        self.assertEqual([config.chunk_size for config in self.opened], [256, 512, 1024])
        self.assertEqual({config.chunk_overlap for config in self.opened}, {50})
        self.assertEqual((summary["comparisons_count"], summary["retrieval_count"], summary["errors"]),
                         (5, 10, 0))
        self.assertEqual(summary["state"], "complete")
        self.assertEqual({run["collection"] for run in summary["runs"] if run["chunk_size"] == 512},
                         {"test-collection-512"})
        self.assertEqual([query[1] for query in self.queries],
                         [case["user_input"] for _ in range(5) for case in self.cases])

    def test_artifacts_record_text_sources_distances_identity_and_latency(self):
        summary = self.run_comparisons()
        manifest, rows = self.read_run(summary, "tokens-512-topk-3")
        self.assertEqual(manifest["corpus"]["corpus_sha256"], hashlib.sha256(self.corpus).hexdigest())
        self.assertEqual(manifest["corpus"]["prepared_manifest"]["revision"], "test-rev")
        self.assertEqual(manifest["dependencies"]["tokenizers"], "fixture")
        self.assertEqual(manifest["model"]["digest"], "simulated-model-digest")
        self.assertEqual(manifest["config"]["tokenizer_revision"], self.config.tokenizer_revision)
        self.assertEqual(rows[0]["case_id"], "q1")
        self.assertEqual(rows[0]["reference"], "Gold only")
        self.assertEqual(rows[0]["retrieved_contexts"], ["Trecho original com acentuação."] * 3)
        chunk = rows[0]["retrieved_chunks"][0]
        self.assertEqual(chunk["source"], "original.txt")
        self.assertEqual(chunk["metadata"]["source_aliases"], ["original.txt", "duplicate.txt"])
        self.assertEqual(chunk["distance"], 0.125)
        self.assertEqual(chunk["distance_metric"], "cosine")
        self.assertGreaterEqual(rows[0]["latency_seconds"], 0)
        self.assertEqual(rows[0]["config"]["chunk_unit"], "tokens")

    def test_opt_in_build_constructs_each_size_once_and_build_failure_is_logged(self):
        built = []

        def build(config):
            built.append(config.chunk_size)
            if config.chunk_size == 512:
                raise RuntimeError("fixture: build interrupted")
            return {"model_digest": "simulated-model-digest"}

        with patch("app.retrieval.experiments.build_index", side_effect=build):
            summary = self.run_comparisons(build=True)
        self.assertEqual(built, [256, 512, 1024])
        self.assertEqual([config.chunk_size for config in self.opened], [256, 1024])
        self.assertEqual(summary["errors"], 6)
        manifest, rows = self.read_run(summary, "tokens-512-topk-5")
        self.assertEqual(manifest["error_stage"], "build")
        self.assertEqual(manifest["config"]["tokenizer_id"], self.config.tokenizer_id)
        self.assertEqual(manifest["corpus"]["selection"], "all")
        self.assertEqual({row["error_stage"] for row in rows}, {"build"})
        self.assertEqual(rows[0]["retrieved_chunks"], [])
        self.assertIsNone(rows[0]["latency_seconds"])
        self.assertIsNone(rows[0]["model"]["digest"])
        self.assertIn("build interrupted", rows[0]["error"])

    def test_open_failure_preserves_case_errors_and_continues_other_sizes(self):
        self.open_failure = 512
        summary = self.run_comparisons()
        self.assertEqual(summary["errors"], 6)
        manifest, rows = self.read_run(summary, "tokens-512-topk-1")
        self.assertEqual(manifest["status"], "error")
        self.assertEqual(manifest["error_stage"], "open")
        self.assertEqual([row["case_id"] for row in rows], ["q1", "q2"])
        self.assertEqual(rows[0]["error_type"], "RuntimeError")
        manifest, rows = self.read_run(summary, "tokens-1024-topk-3")
        self.assertEqual(manifest["status"], "success")
        self.assertEqual(rows[0]["status"], "success")

    def test_query_failure_is_timed_and_does_not_skip_later_cases(self):
        self.query_failure = "Primeira pergunta?"
        summary = self.run_comparisons()
        self.assertEqual(summary["errors"], 5)
        _, rows = self.read_run(summary, "tokens-256-topk-3")
        self.assertEqual(rows[0]["error_stage"], "retrieve")
        self.assertGreaterEqual(rows[0]["latency_seconds"], 0)
        self.assertEqual(rows[1]["status"], "success")

    def test_invalid_protocol_settings_fail_before_creating_artifacts(self):
        invalid = [replace(self.config, chunk_unit="characters"),
                   replace(self.config, chunk_overlap=256)]
        for config in invalid:
            with self.subTest(config=config), self.assertRaises(ValueError):
                run_comparisons(config, self.cases, self.root / "results")
        for kwargs in ({"fixed_top_k": 2}, {"fixed_top_k": True}, {"fixed_chunk_size": 300}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.run_comparisons(**kwargs)
        self.assertFalse((self.root / "results").exists())

    def test_optional_baselines_still_vary_one_factor_and_keep_overlap_constant(self):
        self.config = replace(self.config, chunk_overlap=100)
        summary = self.run_comparisons(fixed_top_k=1, fixed_chunk_size=256)
        self.assertEqual([(run["chunk_size"], run["top_k"]) for run in summary["runs"]],
                         [(256, 1), (512, 1), (1024, 1), (256, 3), (256, 5)])
        self.assertEqual({config.chunk_overlap for config in self.opened}, {100})

    def test_repeated_execution_creates_separate_output_directories(self):
        first = self.run_comparisons()
        first_manifest = (Path(first["output_dir"]) / "manifest.json").read_bytes()
        second = self.run_comparisons()
        self.assertNotEqual(first["output_dir"], second["output_dir"])
        self.assertEqual((Path(first["output_dir"]) / "manifest.json").read_bytes(), first_manifest)


if __name__ == "__main__":
    unittest.main()
