"""Embedding protocol and configuration checks that never require Ollama."""

from dataclasses import replace
import io
import json
import math
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from app.retrieval.config import RetrievalConfig
from app.retrieval.embeddings import EmbeddingError, OllamaEmbeddings


class RetrievalConfigTests(unittest.TestCase):
    def test_positive_integer_fields_reject_booleans_and_invalid_numbers(self):
        for field in ("dimension", "chunk_size", "top_k", "batch_size"):
            for value in (True, False, 0, -1, 1.5, "2"):
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    RetrievalConfig(**{field: value})

    def test_overlap_must_be_integer_smaller_than_chunk_size(self):
        for value in (True, False, -1, 10, 11, 1.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                RetrievalConfig(chunk_size=10, chunk_overlap=value)
        self.assertEqual(RetrievalConfig(chunk_size=10, chunk_overlap=0).chunk_overlap, 0)
        self.assertEqual(RetrievalConfig(chunk_size=10, chunk_overlap=9).chunk_overlap, 9)

    def test_timeout_url_and_embedding_profile_are_validated(self):
        for timeout in (0, -1, math.nan, math.inf):
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                RetrievalConfig(timeout_seconds=timeout)
        for url in ("localhost:11435", "file:///tmp/ollama", "http://localhost:11435/v1"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                RetrievalConfig(base_url=url)
        for kwargs in ({"model": " "}, {"profile": "unknown"}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                RetrievalConfig(**kwargs)

    def test_top_k_and_operational_settings_do_not_change_index_parameters(self):
        original = RetrievalConfig()
        operational = replace(original, top_k=1, batch_size=1, timeout_seconds=20,
                              base_url="http://localhost:11436")
        self.assertEqual(original.index_parameters(), operational.index_parameters())
        for kwargs in ({"model": "another-model"}, {"dimension": 3}, {"profile": "plain"},
                       {"chunk_size": 1400}, {"chunk_overlap": 100}):
            with self.subTest(kwargs=kwargs):
                self.assertNotEqual(original.index_parameters(),
                                    replace(original, **kwargs).index_parameters())

    def test_environment_uses_retrieval_settings_independently_of_generator_and_evaluator(self):
        unrelated = {"OLLAMA_MODEL": "generator", "EVAL_EMBEDDING_MODEL": "judge-embedder"}
        with patch.dict(os.environ, unrelated, clear=True), patch("dotenv.load_dotenv"):
            baseline = RetrievalConfig.from_env()
        self.assertEqual(baseline.model, "embeddinggemma:300m")
        chosen = {**unrelated, "RETRIEVAL_EMBEDDING_MODEL": "retrieval-model",
                  "RETRIEVAL_EMBEDDING_DIMENSION": "3", "RETRIEVAL_EMBEDDING_PROFILE": "plain",
                  "RETRIEVAL_CHUNK_SIZE": "100", "RETRIEVAL_CHUNK_OVERLAP": "10",
                  "RETRIEVAL_TOP_K": "2", "RETRIEVAL_BATCH_SIZE": "4", "RETRIEVAL_CORPUS_SELECTION": "all"}
        with patch.dict(os.environ, chosen, clear=True), patch("dotenv.load_dotenv"):
            configured = RetrievalConfig.from_env()
        self.assertEqual((configured.model, configured.dimension, configured.profile),
                         ("retrieval-model", 3, "plain"))
        self.assertEqual((configured.chunk_size, configured.chunk_overlap, configured.top_k, configured.batch_size),
                         (100, 10, 2, 4))
        self.assertIsNone(configured.corpus_selection)


class OllamaEmbeddingsTests(unittest.TestCase):
    def setUp(self):
        self.config = RetrievalConfig(dimension=3)
        self.embedder = OllamaEmbeddings(self.config)

    def response(self, value):
        return io.BytesIO(json.dumps(value).encode("utf-8"))

    def test_document_prefix_and_truncation_policy_are_sent_to_embed_endpoint(self):
        chunks = [SimpleNamespace(title="Water", text="Safe water matters."),
                  SimpleNamespace(title="", text="A second passage.")]
        vectors = [[1, 0, 0], [0, 1, 0]]
        with patch("app.retrieval.embeddings.urlopen", return_value=self.response({"embeddings": vectors})) as request:
            self.assertEqual(self.embedder.documents(chunks), vectors)
        args, kwargs = request.call_args
        self.assertEqual(args[0].full_url, "http://localhost:11435/api/embed")
        self.assertEqual(args[0].get_method(), "POST")
        body = json.loads(args[0].data)
        self.assertEqual(body["model"], self.config.model)
        self.assertIs(body["truncate"], False)
        self.assertEqual(body["input"], ["title: Water | text: Safe water matters.",
                                          "title: none | text: A second passage."])
        self.assertEqual(kwargs["timeout"], self.config.timeout_seconds)

    def test_query_prefix_and_plain_profile(self):
        with patch.object(self.embedder, "_request", return_value={"embeddings": [[1, 0, 0]]}) as request:
            self.assertEqual(self.embedder.query("Where is water?"), [1, 0, 0])
        self.assertEqual(request.call_args.args[1]["input"],
                         ["task: search result | query: Where is water?"])
        plain = OllamaEmbeddings(replace(self.config, profile="plain"))
        with patch.object(plain, "_request", return_value={"embeddings": [[1, 0, 0]]}) as request:
            plain.documents([SimpleNamespace(title="Ignored", text="passage")])
            self.assertEqual(request.call_args.args[1]["input"], ["passage"])
            plain.query("question")
            self.assertEqual(request.call_args.args[1]["input"], ["question"])

    def test_invalid_embedding_count_dimension_and_values_are_rejected(self):
        invalid = (None, [], [[1, 0, 0], [0, 1, 0]], [[1, 0]], [[0, 0, 0]],
                   [[True, 0, 0]], [["1", 0, 0]], [[math.nan, 0, 0]],
                   [[math.inf, 0, 0]], [[-math.inf, 0, 0]])
        for vectors in invalid:
            with self.subTest(vectors=vectors):
                with patch.object(self.embedder, "_request", return_value={"embeddings": vectors}):
                    with self.assertRaises(EmbeddingError):
                        self.embedder.query("question")

    def test_empty_embedding_batch_or_text_is_rejected_before_http(self):
        for texts in ([], [""], [" "], [None], ["valid", ""]):
            with self.subTest(texts=texts):
                with patch.object(self.embedder, "_request") as request:
                    with self.assertRaises(ValueError):
                        self.embedder._embed(texts)
                    request.assert_not_called()

    def test_connection_timeout_http_and_invalid_json_fail_with_context(self):
        errors = [URLError("connection refused"), TimeoutError("timed out"),
                  OSError("network unavailable"),
                  HTTPError("http://localhost:11435/api/embed", 400, "Bad request", {},
                            io.BytesIO(b"input exceeds context length"))]
        for error in errors:
            with self.subTest(error=type(error).__name__):
                with patch("app.retrieval.embeddings.urlopen", side_effect=error):
                    with self.assertRaises(EmbeddingError) as raised:
                        self.embedder.query("question")
                self.assertIn("/api/embed", str(raised.exception))
                if isinstance(error, HTTPError):
                    self.assertIn("400", str(raised.exception))
                    self.assertIn("input exceeds context length", str(raised.exception))
        with patch("app.retrieval.embeddings.urlopen", return_value=io.BytesIO(b"not json")):
            with self.assertRaises(EmbeddingError):
                self.embedder.query("question")
        for result in (None, [], 1, "unexpected", True):
            with self.subTest(result=result):
                with patch("app.retrieval.embeddings.urlopen", return_value=self.response(result)):
                    with self.assertRaises(EmbeddingError):
                        self.embedder.query("question")

    def test_model_digest_uses_exact_name_and_supports_latest_alias(self):
        tags = {"models": [{"name": "another-model", "digest": "wrong"},
                           {"name": self.config.model, "digest": "expected"}]}
        with patch.object(self.embedder, "_request", return_value=tags) as request:
            self.assertEqual(self.embedder.model_digest(), "expected")
        request.assert_called_once_with("/api/tags")
        untagged = OllamaEmbeddings(replace(self.config, model="model"))
        with patch.object(untagged, "_request", return_value={"models": [{"model": "model:latest", "digest": "latest"}]}):
            self.assertEqual(untagged.model_digest(), "latest")
        for tags in ({"models": []}, {"models": [{"name": self.config.model}]},
                     {"models": [{"name": "wrong", "digest": "wrong"}]}):
            with self.subTest(tags=tags):
                with patch.object(self.embedder, "_request", return_value=tags):
                    with self.assertRaises(EmbeddingError):
                        self.embedder.model_digest()


if __name__ == "__main__":
    unittest.main()
