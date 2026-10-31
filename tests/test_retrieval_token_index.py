"""Token indexes use real Chroma and a local tokenizer, simulated embeddings."""

from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

import test_retrieval_index as fixtures
from test_retrieval_tokens import local_tokenizer

from app.retrieval.config import RetrievalConfig
from app.retrieval.embeddings import EmbeddingError, OllamaEmbeddings
from app.retrieval.index import build_index, Retriever, RetrievalIndexError


class TokenIndexTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.RetrievalIndexTests("runTest")
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.config = replace(self.fixture.config, chunk_unit="tokens", chunk_size=8,
                              chunk_overlap=2, tokenizer_id="test/local", tokenizer_revision="a" * 40)
        directory = self.config.data_dir / "tokenizer/test--local" / self.config.tokenizer_revision
        directory.mkdir(parents=True)
        (directory / "tokenizer.json").write_text(local_tokenizer().backend.to_str(), encoding="utf-8")
        self.embedder = fixtures.FakeEmbedder(self.config)

    def test_token_build_resume_retrieve_and_top_k_share_identity(self):
        built = build_index(self.config, self.embedder)
        retriever = Retriever(self.config, self.embedder)
        row = retriever.retrieve("water", top_k=1)[0]
        self.assertEqual(row["document_id"], "doc-water")
        self.assertEqual(row["metadata"]["chunk_unit"], "tokens")
        self.assertLessEqual(row["metadata"]["token_count"], 8)
        self.assertEqual(row["metadata"]["tokenizer"]["sha256"],
                         built["parameters"]["tokenizer"]["sha256"])
        self.assertEqual(build_index(self.config, self.embedder)["embedded_this_run"], 0)
        changed = replace(self.config, top_k=1)
        self.assertEqual(Retriever(changed, fixtures.FakeEmbedder(changed)).collection_name,
                         retriever.collection_name)
        for changes in ({"chunk_unit": "characters"}, {"chunk_size": 9}, {"chunk_overlap": 1}):
            config = replace(self.config, **changes)
            with self.subTest(changes=changes), self.assertRaises(RetrievalIndexError):
                Retriever(config, fixtures.FakeEmbedder(config))

    def test_token_index_reopens_in_a_fresh_process_with_same_tokenizer(self):
        built = build_index(self.config, self.embedder)
        script = """
import json, sys
from pathlib import Path
sys.path.insert(0, 'tests')
from test_retrieval_index import FakeEmbedder
from app.retrieval.config import RetrievalConfig
from app.retrieval.index import Retriever
c = RetrievalConfig(data_dir=Path(sys.argv[1]), index_dir=Path(sys.argv[2]),
    corpus_selection=None, dimension=3, chunk_unit='tokens', chunk_size=8,
    chunk_overlap=2, tokenizer_id='test/local', tokenizer_revision='a'*40)
r = Retriever(c, FakeEmbedder(c))
print(json.dumps({'collection': r.collection_name, 'row': r.retrieve('water', top_k=1)[0]}))
"""
        result = subprocess.run([sys.executable, "-c", script, str(self.config.data_dir),
                                 str(self.config.index_dir)], cwd=fixtures.ROOT, capture_output=True,
                                text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        output = json.loads(result.stdout)
        self.assertEqual(output["collection"], built["collection"])
        self.assertEqual(output["row"]["document_id"], "doc-water")

    def test_tokenizer_revision_library_and_hash_are_index_parameters(self):
        identity = self.config.index_parameters()
        self.assertEqual(identity["tokenizer"]["library_version"], "0.22.2")
        other = replace(self.config, tokenizer_revision="b" * 40)
        directory = other.data_dir / "tokenizer/test--local" / other.tokenizer_revision
        directory.mkdir(parents=True)
        (directory / "tokenizer.json").write_text(local_tokenizer().backend.to_str(), encoding="utf-8")
        self.assertNotEqual(identity, other.index_parameters())
        self.assertEqual(identity, replace(self.config, top_k=99).index_parameters())

    def test_interrupted_token_build_is_unqueryable_and_resumes_existing_chunks(self):
        class FailingEmbedder(fixtures.FakeEmbedder):
            def documents(self, chunks):
                if self.document_calls:
                    raise RuntimeError("simulated interruption")
                return super().documents(chunks)

        with self.assertRaisesRegex(RuntimeError, "simulated interruption"):
            build_index(self.config, FailingEmbedder(self.config))
        with self.assertRaises(RetrievalIndexError):
            Retriever(self.config, self.embedder)
        resumed = build_index(self.config, self.embedder)
        self.assertEqual(resumed["reused_chunks"], 1)
        self.assertEqual(resumed["embedded_this_run"], resumed["chunk_count"] - 1)
        self.assertEqual(Retriever(self.config, self.embedder).collection.count(), resumed["chunk_count"])


class TokenInputLimitTests(unittest.TestCase):
    def test_input_limit_counts_title_prefix_and_special_tokens_before_http(self):
        config = RetrievalConfig(chunk_unit="tokens", chunk_size=8, chunk_overlap=2,
                                 embedding_max_tokens=12, dimension=3)
        embedder = OllamaEmbeddings(config)
        embedder._tokenizer = local_tokenizer()
        with patch.object(embedder, "_request") as request:
            with self.assertRaisesRegex(EmbeddingError, "prefixo/título/especiais"):
                embedder.documents([type("Chunk", (), {"title": "um dois tres quatro cinco seis",
                                                        "text": "sete oito nove dez"})()])
            request.assert_not_called()
        with patch.object(embedder, "_request", return_value={"embeddings": [[1, 0, 0]]}) as request:
            embedder.query("um")
            self.assertIs(request.call_args.args[1]["truncate"], False)
        with patch.object(embedder, "_request") as request:
            with self.assertRaises(EmbeddingError):
                embedder.query("palavra " * 20)
            request.assert_not_called()

    def test_default_environment_is_tokens_and_legacy_environment_keeps_characters(self):
        with patch.dict(os.environ, {}, clear=True), patch("dotenv.load_dotenv"):
            config = RetrievalConfig.from_env()
        self.assertEqual((config.chunk_unit, config.chunk_size, config.chunk_overlap), ("tokens", 512, 50))
        with patch.dict(os.environ, {"RETRIEVAL_CHUNK_SIZE": "1200"}, clear=True), patch("dotenv.load_dotenv"):
            config = RetrievalConfig.from_env()
        self.assertEqual((config.chunk_unit, config.chunk_size, config.chunk_overlap),
                         ("characters", 1200, 200))
        self.assertEqual(RetrievalConfig().chunk_unit, "characters")


if __name__ == "__main__":
    unittest.main()
