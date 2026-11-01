"""Data preparation and chunking tests; only network downloads are simulated."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from app.retrieval.chunking import chunk_documents
from app.retrieval.data import (
    CORPUS_FILENAME, DEFAULT_REVISION, QUESTIONS_FILENAME,
    Document, load_documents, normalize_text, prepare_dataset,
)

FIXTURE_REVISION = "a" * 40


class DatasetPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.data_dir = Path(self.temporary.name)
        self.reference_map = {
            "Lei nº 9.250": [{"file": "Lei nº 9.250.txt", "artigos": [{"artigo": "25"}]}]
        }
        self.corpus = [
            {"filename": "Lei nº 9.250.txt", "filedata": "  Art. 25\r\n\r\nTributação: R$ 30.639,90.  "},
            {"filename": "Outra origem.txt", "filedata": "Art. 25\n\nTributac\u0327a\u0303o: R$ 30.639,90."},
            {"filename": "Decreto.txt", "filedata": "§ 1º A isenção será mantida."},
            {"filename": "Vazio.txt", "filedata": "\r\n  "},
            {"filename": "Inválido.txt", "filedata": None},
            {"filedata": "Sem fonte"},
            "not an object",
        ]
        self.questions = [{
            "question_number": "001", "question_summary": "OBRIGATORIEDADE",
            "question_text": "  Quem deve declarar?\r\nEm 2024?  ",
            "answer": ["RESPOSTA ESPERADA FORA DO CORPUS", "Item 1: valor."],
            "references": ["Lei nº 9.250, art. 25"],
            "all_formatted_references": self.reference_map,
            "linked_questions": ["002"],
        }]

    def download_fixture(self, url, path):
        rows = self.corpus if url.endswith(CORPUS_FILENAME) else self.questions
        path.write_bytes(json.dumps(rows, ensure_ascii=False).encode("utf-8"))

    def prepare(self):
        with patch("app.retrieval.data._download", side_effect=self.download_fixture) as download:
            manifest = prepare_dataset(self.data_dir, revision=FIXTURE_REVISION)
        return manifest, download.call_count

    def test_prepare_normalizes_deduplicates_and_preserves_source_aliases(self):
        manifest, calls = self.prepare()
        documents = load_documents(self.data_dir / "prepared")
        self.assertEqual(calls, 2)
        self.assertEqual(manifest["documents"]["input"], 7)
        self.assertEqual(manifest["documents"]["accepted"], 2)
        self.assertEqual(manifest["documents"]["discarded"], 5)
        self.assertEqual(manifest["documents"]["reasons"], {
            "duplicate_normalized_text": 1, "empty_text": 1,
            "invalid_text_type": 1, "missing_source": 1, "not_an_object": 1,
        })
        law = next(document for document in documents if "Art. 25" in document.text)
        self.assertEqual(law.text, "Art. 25\n\nTributação: R$ 30.639,90.")
        self.assertEqual(law.metadata["source_aliases"], ["Lei nº 9.250.txt", "Outra origem.txt"])
        self.assertEqual(law.source, "Lei nº 9.250.txt")
        self.assertEqual(law.title, "Lei nº 9.250")
        self.assertEqual(law.metadata["revision"], FIXTURE_REVISION)

    def test_questions_keep_original_structured_references_outside_corpus(self):
        self.prepare()
        prepared = self.data_dir / "prepared"
        question = json.loads((prepared / "questions.jsonl").read_text(encoding="utf-8"))
        self.assertEqual(question["case_id"], "brtaxqa_2024_001")
        self.assertEqual(question["user_input"], "Quem deve declarar?\nEm 2024?")
        self.assertEqual(question["reference"], "RESPOSTA ESPERADA FORA DO CORPUS\nItem 1: valor.")
        self.assertEqual(question["all_formatted_references"], self.reference_map)
        for field, value in self.questions[0].items():
            self.assertEqual(question[field], value)
        self.assertNotIn("RESPOSTA ESPERADA", (prepared / "corpus.jsonl").read_text(encoding="utf-8"))

    def test_repeated_preparation_uses_verified_cache_and_identical_artifacts(self):
        first, _ = self.prepare()
        paths = [self.data_dir / "prepared" / name for name in ("corpus.jsonl", "questions.jsonl", "manifest.json")]
        before = [path.read_bytes() for path in paths]
        second, calls = self.prepare()
        self.assertEqual(calls, 0)
        self.assertEqual(first, second)
        self.assertEqual(before, [path.read_bytes() for path in paths])
        self.assertEqual(len(first["corpus_sha256"]), 64)
        self.assertEqual(len(first["fingerprint"]), 64)

    def test_modified_raw_cache_fails_checksum_verification(self):
        self.prepare()
        cached = self.data_dir / "raw" / FIXTURE_REVISION / CORPUS_FILENAME
        cached.write_text("[]", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            prepare_dataset(self.data_dir, revision=FIXTURE_REVISION)

    def test_official_revision_rejects_unverified_fixture_bytes(self):
        with patch("app.retrieval.data._download", side_effect=self.download_fixture):
            with self.assertRaisesRegex(ValueError, "Official pinned dataset SHA256 mismatch"):
                prepare_dataset(self.data_dir, revision=DEFAULT_REVISION)

    def test_custom_local_files_are_marked_as_local_without_claiming_official_origin(self):
        local = self.data_dir / "supplied"
        local.mkdir()
        self.download_fixture(CORPUS_FILENAME, local / CORPUS_FILENAME)
        self.download_fixture(QUESTIONS_FILENAME, local / QUESTIONS_FILENAME)
        manifest = prepare_dataset(self.data_dir, revision=FIXTURE_REVISION, raw_dir=local)
        for source in manifest["sources"]:
            self.assertEqual(source["acquisition"], "local_file")
            self.assertIsNone(source["url"])
            self.assertTrue(Path(source["local_path"]).is_file())

    def test_missing_answers_remain_unavailable_without_invented_reference(self):
        self.questions = [{"question_number": "002", "question_text": "Pergunta sem resposta"}]
        self.prepare()
        question = json.loads((self.data_dir / "prepared" / "questions.jsonl").read_text(encoding="utf-8"))
        self.assertIsNone(question["reference"])
        self.assertEqual(question["references"], [])

    def test_mutable_revision_and_unusable_corpus_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "pinned"):
            prepare_dataset(self.data_dir, revision="main")
        self.corpus = [{"filename": "Vazio.txt", "filedata": " "}]
        with self.assertRaisesRegex(ValueError, "no usable documents"):
            self.prepare()

    def test_conflicting_question_identifiers_are_not_silently_replaced(self):
        self.questions.append({**self.questions[0], "question_text": "Outra pergunta"})
        with self.assertRaisesRegex(ValueError, "Conflicting question"):
            self.prepare()

    def test_document_ids_are_stable_when_source_records_reordered(self):
        self.prepare()
        before = {document.text: document.id for document in load_documents(self.data_dir / "prepared")}
        other_dir = self.data_dir / "reordered"
        self.corpus.reverse()
        with patch("app.retrieval.data._download", side_effect=self.download_fixture):
            prepare_dataset(other_dir, revision=FIXTURE_REVISION)
        after = {document.text: document.id for document in load_documents(other_dir / "prepared")}
        self.assertEqual(before, after)


class ChunkingTests(unittest.TestCase):
    def document(self, text):
        return Document("doc_id", text, "Lei nº 9.250.txt", "Lei nº 9.250", {"source_aliases": ["Lei nº 9.250.txt"]})

    def test_offsets_coverage_sources_and_hard_character_limit(self):
        text = ("Art. 1º A tributação preserva os números 30.639,90.\n\n" * 12) + "Fim."
        document = self.document(text)
        chunks = list(chunk_documents([document], chunk_size=90, chunk_overlap=15))
        reconstructed = chunks[0].text
        for index, chunk in enumerate(chunks):
            start, end = chunk.metadata["start_char"], chunk.metadata["end_char"]
            self.assertEqual(chunk.text, text[start:end])
            self.assertLessEqual(len(chunk.text), 90)
            self.assertEqual(chunk.source, document.source)
            self.assertEqual(chunk.document_id, document.id)
            self.assertEqual(chunk.metadata["source_aliases"], document.metadata["source_aliases"])
            if index:
                self.assertEqual(start, chunks[index - 1].metadata["end_char"] - 15)
                reconstructed += chunk.text[15:]
        self.assertEqual(reconstructed, text)
        self.assertEqual(chunks[-1].metadata["end_char"], len(text))

    def test_zero_overlap_covers_each_character_exactly_once(self):
        text = "Antes. Depois.\n\nOutro artigo. " * 5
        chunks = list(chunk_documents([self.document(text)], chunk_size=30, chunk_overlap=0))
        self.assertEqual("".join(chunk.text for chunk in chunks), text)
        self.assertEqual(len({chunk.id for chunk in chunks}), len(chunks))

    def test_long_unbroken_text_and_extreme_valid_overlap_make_progress(self):
        text = "á" * 25
        for size, overlap in ((8, 0), (8, 7), (1, 0)):
            with self.subTest(size=size, overlap=overlap):
                chunks = list(chunk_documents([self.document(text)], size, overlap))
                self.assertEqual(chunks[-1].metadata["end_char"], len(text))
                self.assertTrue(all(0 < len(chunk.text) <= size for chunk in chunks))
                self.assertEqual(len({chunk.id for chunk in chunks}), len(chunks))

    def test_natural_boundaries_and_repeatable_configuration_specific_ids(self):
        document = self.document("Artigo um.\n\nArtigo dois e seu conteúdo.\n\nArtigo três.")
        chunks = list(chunk_documents([document], 42, 2))
        self.assertTrue(chunks[0].text.endswith("\n\n"))
        self.assertEqual(chunks, list(chunk_documents([document], 42, 2)))
        other = list(chunk_documents([document], 43, 2))
        self.assertNotEqual(chunks[0].id, other[0].id)

    def test_invalid_parameters_are_rejected(self):
        for size, overlap in ((0, 0), (-1, 0), (5, -1), (5, 5), (5, 6), (True, 0), (5, False), (5.5, 0), (5, 1.5)):
            with self.subTest(size=size, overlap=overlap):
                with self.assertRaises(ValueError):
                    list(chunk_documents([self.document("texto")], size, overlap))

    def test_normalization_preserves_paragraphs_accents_and_numbers(self):
        self.assertEqual(normalize_text(" \r\nArt. 1º\r\n\r\nAc\u0327a\u0303o: R$ 1.234,56.\r "),
                         "Art. 1º\n\nAção: R$ 1.234,56.")


if __name__ == "__main__":
    unittest.main()
