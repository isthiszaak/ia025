"""CARF preparation tests using synthetic records; no service or real embeddings."""

import json
from pathlib import Path
import tempfile
import unittest

from app.retrieval.data import (
    CARF_FILENAME, CORPUS_FILENAME, QUESTIONS_FILENAME,
    load_documents, prepare_dataset,
)


class CarfPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.raw = self.root / "raw"
        self.raw.mkdir()
        self.write(CORPUS_FILENAME, [
            {"filename": "Lei.txt", "filedata": "Artigo 1º.\r\n\r\nTexto da lei."},
            {"filename": "Nome alternativo.txt", "filedata": "Artigo 1º.\n\nTexto da lei."},
        ])
        self.write(QUESTIONS_FILENAME, [{
            "question_number": "001", "question_text": "Qual a regra?",
            "answer": ["GABARITO EXCLUSIVO FORA DO INDICE"],
            "all_formatted_references": {"Lei": [{"file": "Nome alternativo.txt"}]},
        }])
        self.write(CARF_FILENAME, [
            {"filename": "10070000217200724_6764220.txt", "filedata": "Decisão\r\n\r\nAção fiscal."},
            {"filename": "10070000217200724_6764221.txt", "filedata": "Decisa\u0303o\n\nAc\u0327a\u0303o fiscal."},
        ])

    def write(self, filename, rows):
        (self.raw / filename).write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")

    def prepare(self, **kwargs):
        return prepare_dataset(self.root / "data", revision="b" * 40, raw_dir=self.raw, **kwargs)

    def test_carf_opt_in_normalizes_deduplicates_and_preserves_case_origins(self):
        manifest = self.prepare(include_carf=True)
        documents = load_documents(self.root / "data/prepared")
        self.assertEqual(manifest["documents"]["input"], 4)
        self.assertEqual(manifest["documents"]["accepted"], 2)
        self.assertEqual(manifest["documents"]["reasons"], {"duplicate_normalized_text": 2})
        carf = next(document for document in documents if document.metadata["document_type"] == "case_law")
        self.assertEqual(carf.text, "Decisão\n\nAção fiscal.")
        self.assertEqual(carf.metadata["case_year"], 2023)
        self.assertEqual(carf.metadata["case_file_ids"], ["10070000217200724_6764220", "10070000217200724_6764221"])
        self.assertEqual([origin["record_index"] for origin in carf.metadata["source_records"]], [0, 1])
        self.assertTrue(all(origin["dataset_file"] == CARF_FILENAME for origin in carf.metadata["source_records"]))
        self.assertIsNone(carf.metadata["dataset_file_url"])
        self.assertIsNone(carf.metadata["origin_repository"])
        self.assertEqual(len(manifest["sources"]), 3)
        self.assertEqual(manifest["excluded_subsets"], ["expected_answers"])
        self.assertNotIn("GABARITO", (self.root / "data/prepared/corpus.jsonl").read_text(encoding="utf-8"))
        self.assertEqual(manifest["reference_audit"]["matched_reference_files"], 1)
        self.assertEqual(manifest["reference_audit"]["questions_with_explicit_carf_file_references"], 0)

    def test_default_preserves_legal_only_corpus_and_does_not_read_carf(self):
        (self.raw / CARF_FILENAME).write_text("invalid CARF JSON", encoding="utf-8")
        manifest = self.prepare()
        self.assertEqual(manifest["documents"]["accepted"], 1)
        self.assertEqual(len(manifest["sources"]), 2)
        self.assertIn("CARF", manifest["excluded_subsets"])

    def test_cross_subset_duplicates_retain_both_origins_and_legal_identifier(self):
        before = self.prepare()
        legal = load_documents(self.root / "data/prepared")[0]
        self.write(CARF_FILENAME, [{"filename": "123_456.txt", "filedata": legal.text}])
        manifest = self.prepare(include_carf=True)
        documents = load_documents(self.root / "data/prepared")
        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0].id, legal.id)
        self.assertEqual(documents[0].source, legal.source)
        self.assertEqual(set(documents[0].metadata["subsets"]), {"legal_documents_v1.1", "carf_2023"})
        self.assertEqual(len(documents[0].metadata["source_records"]), 3)
        self.assertEqual(manifest["documents"]["reasons"]["duplicate_normalized_text_across_subsets"], 1)
        self.assertNotEqual(before["fingerprint"], manifest["fingerprint"])

    def test_unmatched_references_are_reported_without_fabricated_links(self):
        self.write(QUESTIONS_FILENAME, [{"question_text": "Pergunta", "all_formatted_references": [{"file": "Missing.txt"}]}])
        manifest = self.prepare(include_carf=True)
        audit = manifest["reference_audit"]
        self.assertEqual(audit["unmatched_reference_files"], ["Missing.txt"])
        self.assertEqual(audit["matched_reference_files"], 0)
        prepared_question = json.loads((self.root / "data/prepared/questions.jsonl").read_text(encoding="utf-8"))
        self.assertNotIn("relevant_document_ids", prepared_question)

    def test_carf_repeated_preparation_and_cache_integrity(self):
        first = self.prepare(include_carf=True)
        self.assertEqual(first, self.prepare(include_carf=True))
        self.write(CARF_FILENAME, [])
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            self.prepare(include_carf=True)

    def test_invalid_carf_records_are_reported_and_empty_subset_is_rejected(self):
        self.write(CARF_FILENAME, [{"filename": "Empty.txt", "filedata": "  "}, {"filedata": "No source"}, None])
        with self.assertRaisesRegex(ValueError, "CARF corpus contains no usable documents"):
            self.prepare(include_carf=True)
        with self.assertRaisesRegex(ValueError, "boolean"):
            self.prepare(include_carf="yes")


if __name__ == "__main__":
    unittest.main()
