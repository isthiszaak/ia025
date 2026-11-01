"""Exercise the public CLI contract without any network or embedding service."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from app.retrieval.__main__ import main
from app.retrieval.config import RetrievalConfig
from app.retrieval.index import RetrievalIndexError


class RetrievalCliTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config = RetrievalConfig(data_dir=self.root / "data", index_dir=self.root / "index",
                                     corpus_selection=None, chunk_unit="tokens", chunk_size=512,
                                     chunk_overlap=50)
        prepared = self.config.data_dir / "prepared"
        prepared.mkdir(parents=True)
        (prepared / "corpus.jsonl").write_text("original corpus\n", encoding="utf-8")
        (prepared / "manifest.json").write_text('{"dataset":"fixture"}', encoding="utf-8")
        self.config_patch = patch("app.retrieval.__main__.RetrievalConfig.from_env", return_value=self.config)
        self.config_patch.start()
        self.addCleanup(self.config_patch.stop)

    def run_cli(self, args):
        output = io.StringIO()
        with redirect_stdout(output):
            code = main(args)
        return code, json.loads(output.getvalue())

    def test_prepare_carf_is_explicit_and_build_can_use_prepared_corpus(self):
        with patch("app.retrieval.__main__.prepare_dataset", return_value={"fixture": True}) as prepare:
            code, _ = self.run_cli(["prepare", "--include-carf"])
            self.assertEqual(code, 0)
            self.assertIs(prepare.call_args.kwargs["include_carf"], True)
        with patch("app.retrieval.__main__.prepare_dataset") as prepare, patch(
                "app.retrieval.__main__.build_index", return_value={"state": "ready"}) as build:
            code, _ = self.run_cli(["build", "--use-prepared", "--selection", "all",
                                    "--chunk-unit", "tokens", "--chunk-size", "256", "--chunk-overlap", "50"])
            self.assertEqual(code, 0)
            prepare.assert_not_called()
            config = build.call_args.args[0]
            self.assertIsNone(config.corpus_selection)
            self.assertEqual((config.chunk_unit, config.chunk_size, config.chunk_overlap), ("tokens", 256, 50))

    def test_query_failure_is_saved_with_corpus_configuration_and_no_fake_search_time(self):
        with patch("app.retrieval.__main__.Retriever", side_effect=RetrievalIndexError("index missing")):
            code, result = self.run_cli(["query", "Pergunta?", "--top-k", "3", "--output-dir", str(self.root / "runs")])
        self.assertEqual(code, 1)
        destination = Path(result["output_dir"])
        row = json.loads((destination / "retrieval.jsonl").read_text(encoding="utf-8"))
        manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(row["status"], "error")
        self.assertEqual(row["error_stage"], "open")
        self.assertIsNone(row["latency_seconds"])
        self.assertEqual(row["config"]["chunk_size"], 512)
        self.assertEqual(manifest["corpus"]["prepared_manifest"]["dataset"], "fixture")
        self.assertIn("corpus_sha256", manifest["corpus"])
        self.assertIsNone(manifest["index"])

    def test_query_preserves_references_but_only_embeds_question(self):
        cases = self.root / "cases.jsonl"
        cases.write_text(json.dumps({"case_id": "q1", "user_input": "Pergunta?", "reference": "Gabarito",
                                     "all_formatted_references": {"file": ["source.txt"]}}) + "\n", encoding="utf-8")
        with patch("app.retrieval.__main__.Retriever") as factory:
            retriever = factory.return_value
            retriever.collection_name = "collection"
            retriever.manifest = {"model_digest": "simulated"}
            retriever.retrieve.return_value = [{"text": "Documento", "source": "source.txt"}]
            code, result = self.run_cli(["query", "--cases", str(cases), "--output-dir", str(self.root / "runs")])
            retriever.retrieve.assert_called_once_with("Pergunta?")
        self.assertEqual(code, 0)
        row = json.loads((Path(result["output_dir"]) / "retrieval.jsonl").read_text(encoding="utf-8"))
        self.assertEqual(row["reference"], "Gabarito")
        self.assertEqual(row["all_formatted_references"], {"file": ["source.txt"]})
        self.assertEqual(row["retrieved_contexts"], ["Documento"])

    def test_compare_cli_uses_five_setting_protocol_defaults(self):
        cases = self.root / "cases.jsonl"
        cases.write_text('{"case_id":"q1","user_input":"Pergunta?"}\n', encoding="utf-8")
        with patch("app.retrieval.experiments.run_comparisons", return_value={"errors": 0}) as compare:
            code, _ = self.run_cli(["compare", "--cases", str(cases), "--output-dir", str(self.root / "runs"),
                                    "--selection", "all", "--build-indexes"])
        self.assertEqual(code, 0)
        self.assertEqual(compare.call_args.kwargs,
                         {"fixed_top_k": 3, "fixed_chunk_size": 512, "build": True})
        self.assertIsNone(compare.call_args.args[0].corpus_selection)

    def test_console_json_preserves_unicode_on_legacy_windows_encoding(self):
        raw = io.BytesIO()
        console = io.TextIOWrapper(raw, encoding="cp1252")
        with patch("app.retrieval.__main__.Retriever") as factory, redirect_stdout(console):
            retriever = factory.return_value
            retriever.collection_name = "collection"
            retriever.manifest = {}
            retriever.retrieve.return_value = [{"text": "ação \ue5ce 😀", "source": "norma.txt"}]
            self.assertEqual(main(["query", "Pergunta?"]), 0)
        console.flush()
        self.assertEqual(json.loads(raw.getvalue().decode("cp1252"))["retrieved_contexts"], ["ação \ue5ce 😀"])


if __name__ == "__main__":
    unittest.main()
