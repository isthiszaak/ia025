"""Reproducible preparation of the legal-document BR-TaxQA-R baseline.

The public JSON files are used directly; no Hugging Face dataset loader or
evaluation answers participate in the document corpus.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable
import unicodedata
import urllib.error
import urllib.request


DATASET = "unicamp-dl/BR-TaxQA-R"
DEFAULT_REVISION = "9f0f4263928c6152a40f888da3e63375a24bb0f3"
CORPUS_FILENAME = "referred_legal_documents_QA_2024_v1.1.json"
QUESTIONS_FILENAME = "questions_QA_2024_v1.1.json"
CARF_FILENAME = "acordaos_CARF_2023.json"
NORMALIZATION = "NFC+LF+outer-strip-v1"
SCHEMA_VERSION = 1
# Corpus SHA256 is the published Hugging Face LFS OID. Questions are stored as
# a regular Git blob (OID below); SHA256 was computed from its verified bytes.
PINNED_SOURCE_SHA256 = {
    CORPUS_FILENAME: "49ade2d93ed8f7361274bc93a0fa3d6572a2b938aeb1d343494acd2aa23aecd3",
    QUESTIONS_FILENAME: "097abfc4f885ea36e512e6beae39a826b981191aa0b320960828e51ccc081585",
    CARF_FILENAME: "2fca2a7f0bc89c08f1eefb04ee4167a98c9fc170af25c1ac2468098c9eb079bd",
}
PINNED_QUESTIONS_GIT_BLOB = "3c5c41f20795925747a1b3c5d9f23d2be15ed54d"


@dataclass(frozen=True)
class Document:
    """A normalized legal text with its original source filename(s)."""

    id: str
    text: str
    source: str
    title: str
    metadata: dict[str, Any] = field(default_factory=dict)


def normalize_text(text: str) -> str:
    """Normalize Unicode and line endings without flattening text structure."""
    return unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n")).strip()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    _atomic_bytes(path, (_canonical_json(value) + "\n").encode("utf-8"))


def _atomic_bytes(path: Path, content: bytes) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as output:
            temporary = Path(output.name)
            output.write(content)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            for row in rows:
                encoded = (_canonical_json(row) + "\n").encode("utf-8")
                handle.write(encoded)
                digest.update(encoded)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return digest.hexdigest()


def _read_json_array(path: Path) -> list[Any]:
    try:
        content = json.loads(path.read_text(encoding="utf-8-sig"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid UTF-8 JSON dataset file: {path}") from exc
    if not isinstance(content, list):
        raise ValueError(f"Expected a JSON array in {path}")
    return content


def _download(url: str, destination: Path) -> None:
    """Download atomically, leaving no incomplete file in the cache."""
    request = urllib.request.Request(url, headers={"User-Agent": "ia025-data-retrieval/1"})
    temporary = destination.with_suffix(destination.suffix + ".download")
    try:
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
            while block := response.read(1024 * 1024):
                output.write(block)
        _read_json_array(temporary)
        temporary.replace(destination)
    except (OSError, urllib.error.URLError, ValueError) as exc:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Could not download dataset from {url}: {exc}") from exc


def _cached_source(raw_dir: Path, filename: str, revision: str) -> tuple[Path, dict[str, Any]]:
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / filename
    sidecar = path.with_suffix(path.suffix + ".source.json")
    url = f"https://huggingface.co/datasets/{DATASET}/resolve/{revision}/{filename}"
    downloaded = not path.exists()
    if downloaded:
        _download(url, path)
    _read_json_array(path)
    raw_bytes = path.read_bytes()
    identity = {
        "filename": filename, "revision": revision,
        "sha256": _sha256(raw_bytes), "bytes": len(raw_bytes),
    }
    if revision.lower() == DEFAULT_REVISION:
        if identity["sha256"] != PINNED_SOURCE_SHA256[filename]:
            raise ValueError(f"Official pinned dataset SHA256 mismatch: {path}")
        if filename == QUESTIONS_FILENAME:
            git_blob = hashlib.sha1(f"blob {len(raw_bytes)}\0".encode("ascii") + raw_bytes).hexdigest()
            if git_blob != PINNED_QUESTIONS_GIT_BLOB:
                raise ValueError(f"Official pinned dataset Git blob mismatch: {path}")
        identity["integrity"] = "pinned_official_sha256"

    previous = None
    if sidecar.exists():
        previous = json.loads(sidecar.read_text(encoding="utf-8"))
        if any(previous.get(key) != identity[key] for key in ("filename", "revision", "sha256", "bytes")):
            raise ValueError(f"Dataset cache provenance or SHA256 mismatch: {path}. Use a fresh cache directory.")
    if revision.lower() == DEFAULT_REVISION:
        source = {**identity, "url": url, "acquisition": (previous or {}).get("acquisition", "download" if downloaded else "verified_cache")}
    elif previous and "acquisition" in previous:
        source = previous
    elif downloaded:
        source = {**identity, "url": url, "acquisition": "download"}
    else:
        source = {**identity, "url": None, "acquisition": "local_file", "local_path": str(path.resolve())}
    if previous != source:
        _write_json(sidecar, source)
    return path, source


def _report(total: int, accepted: int, discarded: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "input": total,
        "accepted": accepted,
        "discarded": len(discarded),
        "reasons": dict(sorted(Counter(row["reason"] for row in discarded).items())),
        "discarded_records": discarded,
    }


def _prepare_documents(
    rows: list[Any], revision: str, *,
    subset: str = "legal_documents_v1.1", dataset_file: str = CORPUS_FILENAME,
    dataset_file_url: str | None = None,
) -> tuple[list[Document], dict[str, Any]]:
    # Comparing the normalized strings themselves proves duplicates; hashes only
    # provide compact IDs. Alias filenames remain available for source matching.
    by_text: dict[str, dict[str, Any]] = {}
    discarded: list[dict[str, Any]] = []
    for number, row in enumerate(rows):
        problem: str | None = None
        if not isinstance(row, dict):
            problem = "not_an_object"
        elif not isinstance(row.get("filename"), str) or not row["filename"].strip():
            problem = "missing_source"
        elif not isinstance(row.get("filedata"), str):
            problem = "invalid_text_type"
        elif not normalize_text(row["filedata"]):
            problem = "empty_text"
        if problem:
            discarded.append({"record_index": number, "reason": problem})
            continue
        text = normalize_text(row["filedata"])
        source = normalize_text(row["filename"])
        document_id = "doc_" + _sha256(text.encode("utf-8"))
        if text in by_text:
            by_text[text]["aliases"].add(source)
            by_text[text]["record_indices"].append(number)
            discarded.append({
                "record_index": number, "reason": "duplicate_normalized_text",
                "source": source, "canonical_document_id": document_id,
            })
        else:
            by_text[text] = {"id": document_id, "aliases": {source}, "record_indices": [number]}

    documents = []
    for text, value in by_text.items():
        aliases = sorted(value["aliases"])
        source = aliases[0]
        title = source[:-4] if source.lower().endswith(".txt") else source
        source_records = [
            {"dataset_file": dataset_file, "subset": subset, "record_index": index,
             "filename": normalize_text(rows[index]["filename"])}
            for index in value["record_indices"]
        ]
        documents.append(Document(
            id=value["id"], text=text, source=source, title=title,
            metadata={
                "dataset": DATASET, "revision": revision,
                "subset": subset, "subsets": [subset], "source_aliases": aliases,
                "raw_record_indices": value["record_indices"],
                "document_type": "case_law" if subset == "carf_2023" else "legal_document",
                "dataset_file": dataset_file,
                "dataset_file_url": dataset_file_url,
                "source_records": source_records,
                **({"case_year": 2023,
                    # The upstream file provides filenames and text only. Keep
                    # the original case-file identifiers; do not guess a ruling
                    # number or construct an unverified official document URL.
                    "case_file_ids": sorted(alias.removesuffix(".txt") for alias in aliases),
                    "origin_repository": "https://acordaos.economia.gov.br/solr/acordaos2/browse/" if dataset_file_url else None}
                   if subset == "carf_2023" else {}),
            },
        ))
    documents.sort(key=lambda document: document.id)
    for record in discarded:
        record.update({"dataset_file": dataset_file, "subset": subset})
    return documents, _report(len(rows), len(documents), discarded)


def _merge_document_subsets(groups: Iterable[list[Document]]) -> tuple[list[Document], list[dict[str, Any]]]:
    """Deduplicate across subsets without losing any source record or alias."""
    by_text: dict[str, Document] = {}
    duplicates = []
    for documents in groups:
        for document in documents:
            existing = by_text.get(document.text)
            if existing is None:
                by_text[document.text] = document
                continue
            aliases = sorted(set(existing.metadata["source_aliases"]) | set(document.metadata["source_aliases"]))
            subsets = sorted(set(existing.metadata["subsets"]) | set(document.metadata["subsets"]))
            metadata = {**existing.metadata, "source_aliases": aliases, "subsets": subsets,
                        "source_records": existing.metadata["source_records"] + document.metadata["source_records"]}
            for key in ("case_year", "case_file_ids", "origin_repository"):
                if key in document.metadata:
                    metadata[key] = document.metadata[key]
            by_text[document.text] = Document(existing.id, existing.text, existing.source, existing.title, metadata)
            duplicates.append({"reason": "duplicate_normalized_text_across_subsets",
                               "source": document.source, "canonical_document_id": existing.id,
                               "subset": document.metadata["subset"]})
    return sorted(by_text.values(), key=lambda document: document.id), duplicates


def _reference_file_names(value: Any) -> set[str]:
    """Read explicit upstream file links, without inferring relevance labels."""
    names: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "file":
                values = item if isinstance(item, list) else [item]
                names.update(normalize_text(name) for name in values if isinstance(name, str) and name.strip())
            else:
                names.update(_reference_file_names(item))
    elif isinstance(value, list):
        for item in value:
            names.update(_reference_file_names(item))
    return names


def _audit_question_references(questions: list[dict[str, Any]], documents: list[Document]) -> dict[str, Any]:
    aliases = {alias: document for document in documents for alias in document.metadata["source_aliases"]}
    linked_cases = 0
    referenced_files: set[str] = set()
    matched_carf_cases = 0
    for question in questions:
        files = _reference_file_names(question.get("all_formatted_references", question.get("all_formated_references")))
        linked_cases += bool(files)
        referenced_files.update(files)
        matched_carf_cases += any("carf_2023" in aliases[name].metadata["subsets"] for name in files if name in aliases)
    return {"questions_with_explicit_file_references": linked_cases,
            "questions_without_explicit_file_references": len(questions) - linked_cases,
            "distinct_referenced_files": len(referenced_files),
            "matched_reference_files": len(referenced_files & aliases.keys()),
            "unmatched_reference_files": sorted(referenced_files - aliases.keys()),
            "questions_with_explicit_carf_file_references": matched_carf_cases,
            "policy": "Exact upstream file links only; no inferred CARF relevance or answer-quality labels"}


def _answer_text(answer: Any) -> str | None:
    if isinstance(answer, str):
        text = normalize_text(answer)
    elif isinstance(answer, list) and all(isinstance(line, str) for line in answer):
        text = normalize_text("\n".join(answer))
    else:
        return None
    return text or None


def _prepare_questions(rows: list[Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    questions = []
    discarded: list[dict[str, Any]] = []
    seen: dict[str, str] = {}
    for number, row in enumerate(rows):
        if not isinstance(row, dict):
            discarded.append({"record_index": number, "reason": "not_an_object"})
            continue
        raw_question = row.get("question_text")
        if not isinstance(raw_question, str) or not normalize_text(raw_question):
            discarded.append({"record_index": number, "reason": "empty_or_invalid_question"})
            continue
        question = normalize_text(raw_question)
        raw_id = row.get("question_number")
        suffix = str(raw_id).strip() if raw_id is not None else _sha256(question.encode("utf-8"))
        case_id = f"brtaxqa_2024_{suffix}"
        serialized = _canonical_json(row)
        if case_id in seen:
            if seen[case_id] == serialized:
                discarded.append({"record_index": number, "reason": "duplicate_question", "case_id": case_id})
                continue
            raise ValueError(f"Conflicting question records with case ID {case_id}")
        seen[case_id] = serialized
        # Keep every upstream field, including all_formatted_references (a dict
        # of lists), without reducing it to inferred relevance labels.
        prepared = dict(row)
        prepared.update({
            "case_id": case_id, "user_input": question,
            "reference": _answer_text(row.get("answer")),
        })
        prepared.setdefault("references", [])
        questions.append(prepared)
    return questions, _report(len(rows), len(questions), discarded)


def prepare_dataset(
    data_dir: Path,
    revision: str = DEFAULT_REVISION,
    raw_dir: Path | None = None,
    include_carf: bool = False,
) -> dict[str, Any]:
    """Download/cache the pinned baseline and write separate corpus and cases.

    ``raw_dir`` optionally points directly to a directory containing the two
    upstream JSON files (plus CARF when requested). The default revision verifies published pinned hashes;
    custom revisions supplied locally are explicitly marked as local files.
    Existing bytes receive a provenance sidecar; later runs verify its SHA256.
    The default cache is ``data_dir/raw/<revision>``. Prepared artifacts go in
    ``data_dir/prepared`` and are deterministic, including their manifest.
    """
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        raise ValueError("revision must be a pinned 40-character Git commit SHA")
    if not isinstance(include_carf, bool):
        raise ValueError("include_carf must be boolean")
    data_dir = Path(data_dir)
    raw_dir = Path(raw_dir) if raw_dir is not None else data_dir / "raw" / revision
    corpus_path, corpus_source = _cached_source(raw_dir, CORPUS_FILENAME, revision)
    questions_path, questions_source = _cached_source(raw_dir, QUESTIONS_FILENAME, revision)
    documents, document_report = _prepare_documents(
        _read_json_array(corpus_path), revision, dataset_file_url=corpus_source.get("url"))
    questions, question_report = _prepare_questions(_read_json_array(questions_path))
    if not documents:
        raise ValueError("The legal-document corpus contains no usable documents")
    sources = [corpus_source, questions_source]
    subset_reports = {"legal_documents_v1.1": document_report}
    if include_carf:
        carf_path, carf_source = _cached_source(raw_dir, CARF_FILENAME, revision)
        carf_documents, carf_report = _prepare_documents(
            _read_json_array(carf_path), revision, subset="carf_2023", dataset_file=CARF_FILENAME,
            dataset_file_url=carf_source.get("url"))
        if not carf_documents:
            raise ValueError("The CARF corpus contains no usable documents")
        documents, cross_duplicates = _merge_document_subsets([documents, carf_documents])
        subset_reports["carf_2023"] = carf_report
        discarded = document_report["discarded_records"] + carf_report["discarded_records"] + cross_duplicates
        document_report = _report(document_report["input"] + carf_report["input"], len(documents), discarded)
        sources.append(carf_source)
    prepared_dir = data_dir / "prepared"
    prepared_dir.mkdir(parents=True, exist_ok=True)
    corpus_hash = _write_jsonl(prepared_dir / "corpus.jsonl", (asdict(document) for document in documents))
    questions_hash = _write_jsonl(prepared_dir / "questions.jsonl", questions)
    identity = {
        "schema_version": SCHEMA_VERSION, "dataset": DATASET, "revision": revision,
        "subset": "legal_documents_v1.1+carf_2023" if include_carf else "legal_documents_v1.1",
        "normalization": NORMALIZATION,
        "corpus_sha256": corpus_hash,
    }
    manifest = {
        **identity,
        "fingerprint": _sha256(_canonical_json(identity).encode("utf-8")),
        "questions_sha256": questions_hash,
        "sources": sources,
        "documents": document_report,
        "document_subsets": subset_reports,
        "questions": question_report,
        "reference_audit": _audit_question_references(questions, documents),
        "excluded_subsets": ["expected_answers"] if include_carf else ["CARF", "expected_answers"],
        "reference_field": "answer joined with LF; all original fields preserved",
    }
    _write_json(prepared_dir / "manifest.json", manifest)
    return manifest


def load_documents(prepared_dir: Path) -> list[Document]:
    """Read prepared corpus JSONL; fail on corruption or empty documents."""
    path = Path(prepared_dir) / "corpus.jsonl"
    documents = []
    ids = set()
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                document = Document(**row)
                if not all(isinstance(value, str) and value for value in (
                    document.id, document.text, document.source, document.title,
                )) or not isinstance(document.metadata, dict):
                    raise ValueError("invalid document fields")
                if document.id in ids:
                    raise ValueError("duplicate document ID")
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid prepared document in {path}:{number}: {exc}") from exc
            ids.add(document.id)
            documents.append(document)
    if not documents:
        raise ValueError(f"Prepared corpus contains no documents: {path}")
    return documents
