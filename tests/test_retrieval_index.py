"""Real persistent Chroma tests with deterministic, local embedding vectors."""

from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest

import chromadb
from chromadb.api.shared_system_client import SharedSystemClient
from chromadb.config import Settings

from app.retrieval.chunking import chunk_documents
from app.retrieval.config import RetrievalConfig
from app.retrieval.data import Document
from app.retrieval.index import RetrievalIndexError, Retriever, build_index


ROOT = Path(__file__).resolve().parents[1]


class FakeEmbedder:
    """Makes nearest-neighbor ordering predictable without any model service."""

    def __init__(self, config, digest="fake-digest"):
        self.config = config
        self.digest = digest
        self.document_calls = []
        self.query_calls = []

    def model_digest(self):
        return self.digest

    def _vector(self, text):
        axis = 0 if "water" in text.lower() else 1 if "orbit" in text.lower() else 2
        return [1.0 if index == axis else 0.0 for index in range(self.config.dimension)]

    def documents(self, chunks):
        self.document_calls.append([chunk.id for chunk in chunks])
        return [self._vector(chunk.text) for chunk in chunks]

    def query(self, text):
        self.query_calls.append(text)
        return self._vector(text)


class RetrievalIndexTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.config = RetrievalConfig(data_dir=self.root / "data", index_dir=self.root / "index",
                                      corpus_selection=None,
                                      dimension=3, chunk_size=120, chunk_overlap=20,
                                      top_k=2, batch_size=1)
        self.documents = [
            Document("doc-water", "Clean water is available from a protected well.", "water.txt", "Water",
                     {"source_aliases": ["water.txt"], "nested": {"section": 1}}),
            Document("doc-orbit", "The orbit of the shuttle crosses the equator.", "orbit.txt", "Orbit"),
            Document("doc-food", "Nutrition provides energy for healthy growth.", "food.txt", "Nutrition"),
        ]
        self.write_corpus(self.documents)
        self.embedder = FakeEmbedder(self.config)

    def tearDown(self):
        # Chroma has no public close API. Stop its real systems so Windows can
        # remove the temporary SQLite/HNSW files after each integration test.
        try:
            for system in list(SharedSystemClient._identifier_to_system.values()):
                system.stop()
            SharedSystemClient.clear_system_cache()
        finally:
            self.temporary.cleanup()

    def write_corpus(self, documents):
        prepared = self.config.data_dir / "prepared"
        prepared.mkdir(parents=True, exist_ok=True)
        corpus = prepared / "corpus.jsonl"
        corpus.write_bytes(("\n".join(json.dumps(asdict(document), ensure_ascii=False)
                                    for document in documents) + "\n").encode("utf-8"))
        manifest = {"schema_version": 1, "dataset": "test-fixture", "revision": "test-rev",
                    "corpus_sha256": hashlib.sha256(corpus.read_bytes()).hexdigest()}
        (prepared / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def build(self):
        return build_index(self.config, self.embedder)

    def open(self, config=None, embedder=None):
        return Retriever(config or self.config, embedder or self.embedder)

    def client(self):
        return chromadb.PersistentClient(path=str(self.config.index_dir),
                                         settings=Settings(anonymized_telemetry=False))

    def test_build_persists_ids_metadata_cosine_results_and_no_duplicates(self):
        built = self.build()
        expected = list(chunk_documents(self.documents, self.config.chunk_size, self.config.chunk_overlap))
        self.assertEqual(built["state"], "ready")
        self.assertEqual(built["chunk_count"], len(expected))
        self.assertEqual(built["document_count"], len(self.documents))
        self.assertEqual(built["embedded_this_run"], len(expected))
        retriever = self.open()
        self.assertEqual(retriever.collection_name, built["collection"])
        self.assertEqual(retriever.collection.configuration["hnsw"]["space"], "cosine")
        self.assertEqual(set(retriever.collection.get(include=[])["ids"]), {chunk.id for chunk in expected})
        rows = retriever.retrieve(" water ", top_k=99)
        self.assertEqual(len(rows), len(expected))
        self.assertEqual(rows[0]["document_id"], "doc-water")
        self.assertAlmostEqual(rows[0]["distance"], 0, places=5)
        self.assertEqual([row["distance"] for row in rows], sorted(row["distance"] for row in rows))
        self.assertEqual(rows[0]["text"], self.documents[0].text)
        self.assertEqual(rows[0]["source"], "water.txt")
        self.assertEqual(rows[0]["metadata"]["nested"], {"section": 1})
        self.assertEqual(rows[0]["metadata"]["source_aliases"], ["water.txt"])
        for row in rows:
            self.assertTrue(math.isfinite(row["distance"]))
            self.assertTrue({"chunk_id", "document_id", "text", "source", "metadata", "distance"} <= row.keys())
            self.assertEqual(row["distance_metric"], "cosine")
            self.assertEqual(row["text"], next(document.text for document in self.documents
                                                if document.id == row["document_id"])[
                row["metadata"]["start_char"]:row["metadata"]["end_char"]])
        self.assertEqual(self.embedder.query_calls, ["water"])
        calls_before = len(self.embedder.document_calls)
        rerun = self.build()
        self.assertEqual(rerun["collection"], built["collection"])
        self.assertEqual(rerun["embedded_this_run"], 0)
        self.assertEqual(rerun["reused_chunks"], len(expected))
        self.assertEqual(len(self.embedder.document_calls), calls_before)
        self.assertEqual(retriever.collection.count(), len(expected))

    def test_persisted_index_reopens_in_another_process_without_embedding_documents(self):
        built = self.build()
        script = textwrap.dedent("""
            import json
            from pathlib import Path
            import sys
            sys.path.insert(0, sys.argv[3])
            from test_retrieval_index import FakeEmbedder
            from app.retrieval.config import RetrievalConfig
            from app.retrieval.index import Retriever
            config = RetrievalConfig(data_dir=Path(sys.argv[1]), index_dir=Path(sys.argv[2]),
                                     corpus_selection=None,
                                     dimension=3, chunk_size=120, chunk_overlap=20, top_k=1)
            embedder = FakeEmbedder(config)
            retriever = Retriever(config, embedder)
            rows = retriever.retrieve('water')
            assert not embedder.document_calls
            print(json.dumps({'collection': retriever.collection_name, 'rows': rows}))
        """)
        completed = subprocess.run([sys.executable, "-c", script, str(self.config.data_dir),
                                    str(self.config.index_dir), str(ROOT / "tests")],
                                   cwd=ROOT, text=True, capture_output=True, timeout=60, check=False)
        self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
        output = json.loads(completed.stdout.strip().splitlines()[-1])
        self.assertEqual(output["collection"], built["collection"])
        self.assertEqual(output["rows"][0]["document_id"], "doc-water")

    def test_top_k_changes_reuse_existing_index_and_do_not_embed_documents(self):
        built = self.build()
        changed = replace(self.config, top_k=1, batch_size=2)
        other_embedder = FakeEmbedder(changed)
        retriever = self.open(changed, other_embedder)
        self.assertEqual(retriever.collection_name, built["collection"])
        self.assertEqual(len(retriever.retrieve("water")), 1)
        self.assertEqual(len(retriever.retrieve("water", top_k=2)), 2)
        self.assertEqual(other_embedder.document_calls, [])

    def test_invalid_question_or_top_k_is_rejected_before_query_embedding(self):
        self.build()
        retriever = self.open()
        for question in (None, 1, "", " \n"):
            with self.subTest(question=question), self.assertRaises(ValueError):
                retriever.retrieve(question)
        for top_k in (True, False, 0, -1, 1.5, "2"):
            with self.subTest(top_k=top_k), self.assertRaises(ValueError):
                retriever.retrieve("water", top_k=top_k)
        self.assertEqual(self.embedder.query_calls, [])

    def test_changed_model_digest_is_detected_for_an_already_open_retriever(self):
        self.build()
        retriever = self.open()
        self.embedder.digest = "model-updated-after-opening"
        with self.assertRaises(RetrievalIndexError):
            retriever.retrieve("water")
        self.embedder.digest = "fake-digest"
        self.assertEqual(retriever.retrieve("water", top_k=1)[0]["document_id"], "doc-water")

    def test_missing_index_is_reported_without_creating_one(self):
        with self.assertRaises(RetrievalIndexError):
            self.open()
        self.assertFalse(self.config.index_dir.exists())

    def test_missing_corpus_or_prepared_manifest_prevents_build(self):
        prepared = self.config.data_dir / "prepared"
        for filename in ("corpus.jsonl", "manifest.json"):
            path = prepared / filename
            original = path.read_bytes()
            path.unlink()
            try:
                with self.subTest(filename=filename), self.assertRaises(RetrievalIndexError):
                    self.build()
            finally:
                path.write_bytes(original)
        manifest = prepared / "manifest.json"
        original = manifest.read_bytes()
        for invalid in (b"[]", b"null", b"{invalid"):
            manifest.write_bytes(invalid)
            try:
                with self.subTest(manifest=invalid), self.assertRaises(RetrievalIndexError):
                    self.build()
            finally:
                manifest.write_bytes(original)
        self.assertEqual(self.embedder.document_calls, [])
        self.assertFalse(self.config.index_dir.exists())

    def test_corpus_hash_mismatch_is_rejected_and_changed_valid_corpus_needs_build(self):
        built = self.build()
        corpus = self.config.data_dir / "prepared/corpus.jsonl"
        corpus.write_bytes(corpus.read_bytes() + b"\n")
        with self.assertRaises(RetrievalIndexError):
            self.open()
        self.write_corpus(self.documents + [Document("new", "Additional material.", "new.txt", "New")])
        with self.assertRaises(RetrievalIndexError):
            self.open()
        self.assertEqual(self.client().get_collection(built["collection"], embedding_function=None).count(),
                         len(self.documents))

    def test_model_digest_and_all_vector_parameters_require_a_matching_index(self):
        self.build()
        changes = ({"model": "other-model"}, {"dimension": 4}, {"profile": "plain"},
                   {"chunk_size": 100}, {"chunk_overlap": 10})
        for kwargs in changes:
            config = replace(self.config, **kwargs)
            with self.subTest(kwargs=kwargs), self.assertRaises(RetrievalIndexError):
                self.open(config, FakeEmbedder(config))
        with self.assertRaises(RetrievalIndexError):
            self.open(embedder=FakeEmbedder(self.config, digest="different-digest"))
        changed = replace(self.config, chunk_size=100)
        changed_embedder = FakeEmbedder(changed)
        original = self.open()
        alternate = build_index(changed, changed_embedder)
        self.assertNotEqual(alternate["collection"], original.collection_name)
        self.assertEqual(len(self.client().list_collections()), 2)
        self.assertEqual(self.open(changed, changed_embedder).retrieve("water", top_k=1)[0]["document_id"],
                         "doc-water")
        self.assertEqual(original.retrieve("water", top_k=1)[0]["document_id"], "doc-water")

    def test_document_selection_indexes_only_selected_sources_and_has_its_own_collection(self):
        selection_path = self.root / "selection.json"

        def select(ids):
            selection_path.write_text(json.dumps({"revision": "test-rev", "document_ids": ids}),
                                      encoding="utf-8")

        select(["doc-water", "doc-food"])
        selected_config = replace(self.config, corpus_selection=selection_path)
        selected_embedder = FakeEmbedder(selected_config)
        built = build_index(selected_config, selected_embedder)
        self.assertEqual(built["document_count"], 2)
        self.assertEqual(built["chunk_count"], 2)
        self.assertEqual(built["document_ids"], ["doc-food", "doc-water"])
        expected_ids = {chunk.id for chunk in chunk_documents(
            [self.documents[0], self.documents[2]], self.config.chunk_size, self.config.chunk_overlap)}
        self.assertEqual({chunk_id for batch in selected_embedder.document_calls for chunk_id in batch},
                         expected_ids)
        selected_retriever = self.open(selected_config, selected_embedder)
        self.assertEqual({row["document_id"] for row in selected_retriever.retrieve("water", top_k=99)},
                         {"doc-water", "doc-food"})
        select(["doc-orbit"])
        with self.assertRaises(RetrievalIndexError):
            self.open(selected_config, selected_embedder)
        alternate = build_index(selected_config, selected_embedder)
        full = self.build()
        self.assertEqual(len({built["collection"], alternate["collection"], full["collection"]}), 3)
        self.assertEqual((alternate["document_count"], full["document_count"]), (1, 3))
        self.assertEqual(selected_retriever.collection.count(), 2)

    def test_invalid_document_selection_is_rejected_before_creating_an_index(self):
        selection_path = self.root / "selection.json"
        config = replace(self.config, corpus_selection=selection_path)
        invalid = [{}, [], {"document_ids": []},
                   {"document_ids": ["doc-water", "doc-water"]},
                   {"document_ids": ["not-in-corpus"]}, {"document_ids": [1]},
                   {"document_ids": "doc-water"},
                   {"document_ids": ["doc-water"], "revision": "wrong-revision"}]
        for selection in invalid:
            if isinstance(selection, dict):
                selection = {"revision": "test-rev", **selection}
            selection_path.write_text(json.dumps(selection), encoding="utf-8")
            with self.subTest(selection=selection), self.assertRaises(RetrievalIndexError):
                build_index(config, FakeEmbedder(config))
        self.assertFalse(self.config.index_dir.exists())

    def test_incomplete_manifest_and_incomplete_collection_are_not_queryable(self):
        built = self.build()
        manifest_path = Path(built["manifest_path"])
        original = manifest_path.read_bytes()
        for change in ({"state": "building"}, {"chunk_count": 999},
                       {"model_digest": "corrupt"}, {"fingerprint": "corrupt"}):
            manifest = json.loads(original)
            manifest.update(change)
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            try:
                with self.subTest(change=change), self.assertRaises(RetrievalIndexError):
                    self.open()
            finally:
                manifest_path.write_bytes(original)
        for invalid in (b"[]", b"null", b"{invalid"):
            manifest_path.write_bytes(invalid)
            try:
                with self.subTest(manifest=invalid), self.assertRaises(RetrievalIndexError):
                    self.open()
            finally:
                manifest_path.write_bytes(original)
        collection = self.client().get_collection(built["collection"], embedding_function=None)
        collection.modify(metadata={**collection.metadata, "state": "building"})
        with self.assertRaises(RetrievalIndexError):
            self.open()
        collection.modify(metadata={**collection.metadata, "state": "ready"})
        collection.delete(ids=collection.get(include=[])["ids"][:1])
        with self.assertRaises(RetrievalIndexError):
            self.open()

    def test_interrupted_build_remains_unqueryable_and_resumes_without_duplicates(self):
        class FailingEmbedder(FakeEmbedder):
            def documents(self, chunks):
                if self.document_calls:
                    raise RuntimeError("simulated embedding failure")
                return super().documents(chunks)

        with self.assertRaisesRegex(RuntimeError, "simulated embedding failure"):
            build_index(self.config, FailingEmbedder(self.config))
        with self.assertRaises(RetrievalIndexError):
            self.open()
        self.assertEqual(list(self.config.index_dir.glob("*.build.lock")), [])
        resumed = self.build()
        self.assertEqual(resumed["embedded_this_run"], 2)
        self.assertEqual(resumed["reused_chunks"], 1)
        self.assertEqual(self.open().collection.count(), 3)
        self.assertEqual(self.open().retrieve("water", top_k=1)[0]["document_id"], "doc-water")


if __name__ == "__main__":
    unittest.main()
