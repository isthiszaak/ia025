"""Real tokenizer tests with tiny local vocabularies; no model/network required."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors, trainers

from app.retrieval.chunking import chunk_documents
from app.retrieval.data import Document
from app.retrieval.tokenizer import (
    DEFAULT_TOKENIZER, DEFAULT_TOKENIZER_REVISION,
    TextTokenizer, TokenizerError, load_tokenizer,
)


def local_tokenizer(byte_level=False):
    if byte_level:
        backend = Tokenizer(models.BPE(unk_token="[UNK]"))
        backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        backend.train_from_iterator(["tributação ação água 😀 🚀 " * 5], trainers.BpeTrainer(
            vocab_size=257, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=["[UNK]"],
        ))
    else:
        backend = Tokenizer(models.WordLevel({"[UNK]": 0, "<bos>": 1, "<eos>": 2}, unk_token="[UNK]"))
        backend.add_special_tokens(["<bos>", "<eos>"])
        backend.pre_tokenizer = pre_tokenizers.Whitespace()
        backend.post_processor = processors.TemplateProcessing(
            single="<bos> $A <eos>", special_tokens=[("<bos>", 1), ("<eos>", 2)],
        )
    digest = hashlib.sha256(backend.to_str().encode()).hexdigest()
    return TextTokenizer(backend, "test/local", "a" * 40, digest)


class TokenChunkingTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = local_tokenizer()

    def document(self, text):
        return Document("legal-id", text, "Lei nº 1.txt", "Lei nº 1", {
            "source_aliases": ["Lei nº 1.txt", "Cópia.txt"], "revision": "test",
        })

    def chunks(self, text, size=8, overlap=2, tokenizer=None):
        return list(chunk_documents([self.document(text)], size, overlap,
                                   chunk_unit="tokens", tokenizer=tokenizer or self.tokenizer))

    def assert_exact_coverage(self, text, chunks, budget, tokenizer=None):
        covered = 0
        for chunk in chunks:
            metadata = chunk.metadata
            self.assertLessEqual(metadata["start_char"], covered)
            self.assertGreater(metadata["end_char"], covered)
            self.assertEqual(chunk.text, text[metadata["start_char"]:metadata["end_char"]])
            self.assertLessEqual((tokenizer or self.tokenizer).count(chunk.text, False), budget)
            self.assertEqual(metadata["token_count"], (tokenizer or self.tokenizer).count(chunk.text, False))
            self.assertEqual(chunk.document_id, "legal-id")
            self.assertEqual(metadata["source_aliases"], ["Lei nº 1.txt", "Cópia.txt"])
            self.assertEqual(chunk.source, "Lei nº 1.txt")
            covered = metadata["end_char"]
        self.assertEqual(covered, len(text))

    def test_token_windows_keep_text_whitespace_offsets_overlap_and_sources(self):
        text = "  Art. 1º\n\nA tributação preserva R$ 30.639,90.   " * 8 + "\nFim.  "
        chunks = self.chunks(text)
        self.assert_exact_coverage(text, chunks, 8)
        self.assertGreater(len(chunks[0].text), 8)  # token budget is not a character limit
        for previous, current in zip(chunks, chunks[1:]):
            self.assertEqual(previous.metadata["end_token"] - current.metadata["start_token"], 2)
            self.assertEqual(current.metadata["actual_overlap_tokens"], 2)

    def test_zero_overlap_reconstructs_exact_text(self):
        text = "Antes. Depois.\n\nAção R$ 123.456,78! " * 6
        chunks = self.chunks(text, overlap=0)
        self.assertEqual("".join(chunk.text for chunk in chunks), text)
        self.assert_exact_coverage(text, chunks, 8)

    def test_planned_sizes_are_real_token_budgets_with_constant_overlap(self):
        text = "Artigo tributário número um.\n" * 700
        for size in (256, 512, 1024):
            with self.subTest(size=size):
                chunks = self.chunks(text, size=size, overlap=50)
                self.assert_exact_coverage(text, chunks, size)
                self.assertEqual(chunks[0].metadata["token_count"], size)
                self.assertTrue(all(row.metadata["actual_overlap_tokens"] == 50 for row in chunks[1:]))

    def test_unicode_byte_fallback_never_cuts_characters_or_loses_content(self):
        tokenizer = local_tokenizer(byte_level=True)
        text = "Ação água 😀🚀\n\n" * 30
        chunks = self.chunks(text, size=13, overlap=3, tokenizer=tokenizer)
        self.assert_exact_coverage(text, chunks, 13, tokenizer)
        self.assertTrue(all(0 <= row.metadata["actual_overlap_tokens"] <= 3 for row in chunks))
        self.assertTrue(all("�" not in row.text for row in chunks))

    def test_chunk_identity_is_stable_and_distinguishes_tokenizer_and_units(self):
        text = "Artigo um. Artigo dois. Artigo três."
        chunks = self.chunks(text)
        self.assertEqual(chunks, self.chunks(text))
        changed = TextTokenizer(self.tokenizer.backend, "test/local", "b" * 40,
                                self.tokenizer.identity()["sha256"])
        self.assertNotEqual(chunks[0].id, self.chunks(text, tokenizer=changed)[0].id)
        self.assertNotEqual(chunks[0].id, self.chunks(text, size=9)[0].id)
        self.assertNotEqual(chunks[0].id, list(chunk_documents([self.document(text)], 8, 2))[0].id)
        self.assertEqual(chunks[0].metadata["tokenizer"], self.tokenizer.identity())

    def test_special_tokens_counted_only_for_complete_embedding_inputs(self):
        self.assertEqual(self.tokenizer.count("texto simples", False), 2)
        self.assertEqual(self.tokenizer.count("texto simples", True), 4)
        self.tokenizer.backend.enable_truncation(2)
        fresh = TextTokenizer(self.tokenizer.backend, "test/local", "a" * 40, "test-hash")
        self.assertEqual(fresh.count("um dois três quatro cinco", True), 7)

    def test_whitespace_only_text_is_preserved_even_with_zero_content_tokens(self):
        text = " \n\n    "
        chunks = self.chunks(text)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text, text)
        self.assertEqual(chunks[0].metadata["token_count"], 0)

    def test_literal_special_tokens_and_normalizer_ignored_characters_keep_offsets(self):
        text = "\u200b<bos> Ação\u200b <eos>\n\n" * 10 + "\u200b"
        backend = local_tokenizer().backend
        backend.normalizer = normalizers.Replace("\u200b", "")
        tokenizer = TextTokenizer(backend, "test/normalizer", "a" * 40, "fixture")
        chunks = self.chunks(text, size=5, overlap=1, tokenizer=tokenizer)
        self.assert_exact_coverage(text, chunks, 5, tokenizer)
        ignored = self.chunks("\u200b\u200b", size=5, overlap=1, tokenizer=tokenizer)
        self.assertEqual(ignored[0].text, "\u200b\u200b")
        self.assertEqual(ignored[0].metadata["token_count"], 0)

    def test_invalid_unit_missing_tokenizer_and_unrepresentable_budget_are_explicit(self):
        with self.assertRaisesRegex(ValueError, "chunk_unit"):
            list(chunk_documents([self.document("text")], 8, 2, chunk_unit="bytes"))
        with self.assertRaisesRegex(ValueError, "explicit tokenizer"):
            list(chunk_documents([self.document("text")], 8, 2, chunk_unit="tokens"))
        with self.assertRaisesRegex(ValueError, "complete Unicode"):
            self.chunks("😀", size=1, overlap=0, tokenizer=local_tokenizer(byte_level=True))


class TokenizerCacheTests(unittest.TestCase):
    def test_cached_custom_tokenizer_reopens_and_detects_modified_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "test--local" / ("a" * 40) / "tokenizer.json"
            path.parent.mkdir(parents=True)
            path.write_text(local_tokenizer().backend.to_str(), encoding="utf-8")
            first = load_tokenizer(root, "test/local", "a" * 40)
            second = load_tokenizer(root, "test/local", "a" * 40)
            self.assertEqual(first.identity(), second.identity())
            self.assertEqual(first.count("texto simples"), second.count("texto simples"))
            sidecar = json.loads(path.with_name("source.json").read_text(encoding="utf-8"))
            self.assertEqual(sidecar["sha256"], first.identity()["sha256"])
            self.assertEqual(sidecar["acquisition"], "local_file")
            self.assertIsNone(sidecar["url"])
            path.write_bytes(path.read_bytes() + b"\n")
            with self.assertRaisesRegex(TokenizerError, "SHA256 mismatch"):
                load_tokenizer(root, "test/local", "a" * 40)

    def test_official_artifact_hash_is_required_and_mutable_revision_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(ValueError, "pinned"):
                load_tokenizer(root, "test/local", "main")
            path = root / DEFAULT_TOKENIZER.replace("/", "--") / DEFAULT_TOKENIZER_REVISION / "tokenizer.json"
            path.parent.mkdir(parents=True)
            path.write_text(local_tokenizer().backend.to_str(), encoding="utf-8")
            with self.assertRaisesRegex(TokenizerError, "Pinned tokenizer SHA256 mismatch"):
                load_tokenizer(root)


if __name__ == "__main__":
    unittest.main()
