"""Persistent Chroma retrieval, independent of generation and evaluation."""

from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import math
import platform
import sys

from .chunking import chunk_documents
from .config import RetrievalConfig
from .data import load_documents
from .embeddings import OllamaEmbeddings


class RetrievalIndexError(RuntimeError):
    pass


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def _atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _plan(config, embedder):
    prepared = config.data_dir / "prepared"
    corpus = prepared / "corpus.jsonl"
    manifest_file = prepared / "manifest.json"
    if not corpus.is_file() or not manifest_file.is_file():
        raise RetrievalIndexError("Corpus ausente. Execute python -m app.retrieval prepare")
    try:
        dataset = json.loads(manifest_file.read_text(encoding="utf-8"))
        if not isinstance(dataset, dict):
            raise ValueError("manifesto deve ser objeto JSON")
    except (ValueError, OSError) as exc:
        raise RetrievalIndexError(f"Manifesto de dados inválido: {exc}") from exc
    corpus_sha256 = hashlib.sha256(corpus.read_bytes()).hexdigest()
    if dataset.get("corpus_sha256") != corpus_sha256:
        raise RetrievalIndexError("Corpus diverge do manifesto; execute prepare novamente")
    document_ids = None
    if config.corpus_selection is not None:
        try:
            selection = json.loads(config.corpus_selection.read_text(encoding="utf-8"))
            document_ids = selection["document_ids"]
            if (not isinstance(document_ids, list) or not document_ids
                    or any(not isinstance(item, str) or not item for item in document_ids)
                    or len(set(document_ids)) != len(document_ids)):
                raise ValueError("document_ids deve ser uma lista não vazia de IDs únicos")
            if selection.get("revision") != dataset.get("revision"):
                raise ValueError("revisão da seleção difere do corpus preparado")
            available = {document.id for document in load_documents(prepared)}
            if not set(document_ids) <= available:
                raise ValueError("seleção contém documentos ausentes do corpus")
            document_ids = sorted(document_ids)
        except (ValueError, KeyError, TypeError, OSError) as exc:
            raise RetrievalIndexError(f"Seleção de corpus inválida: {exc}") from exc
    identity = {"schema": 1, "corpus_sha256": corpus_sha256, "document_ids": document_ids,
                "parameters": config.index_parameters(), "model_digest": embedder.model_digest()}
    fingerprint = _hash(identity)
    collection_name = "br_taxqa_" + fingerprint[:32]
    return identity, dataset, fingerprint, collection_name


def _client(directory):
    import chromadb
    from chromadb.config import Settings
    return chromadb.PersistentClient(path=str(directory), settings=Settings(anonymized_telemetry=False))


def dependency_versions():
    result = {"python": platform.python_version()}
    for package in ("chromadb", "python-dotenv"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "not installed"
    return result


def build_index(config=None, embedder=None):
    """Build/resume only this configuration; stable IDs make reruns idempotent."""
    config = config or RetrievalConfig.from_env()
    embedder = embedder or OllamaEmbeddings(config)
    identity, dataset, fingerprint, name = _plan(config, embedder)
    documents = load_documents(config.data_dir / "prepared")
    if identity["document_ids"] is not None:
        documents = [document for document in documents if document.id in identity["document_ids"]]
    chunks = list(chunk_documents(documents,
                                  config.chunk_size, config.chunk_overlap))
    if not chunks:
        raise RetrievalIndexError("Corpus sem trechos; nenhum índice foi criado")
    config.index_dir.mkdir(parents=True, exist_ok=True)
    lock_path = config.index_dir / (name + ".build.lock")
    try:
        lock = lock_path.open("x", encoding="utf-8")
    except FileExistsError as exc:
        raise RetrievalIndexError(f"Construção em andamento ou interrompida: {lock_path}. "
                                  "Se não houver processo ativo, remova apenas esse lock.") from exc
    with lock:
        try:
            return _build(config, embedder, identity, dataset, fingerprint, name, chunks)
        finally:
            lock.close()
            lock_path.unlink(missing_ok=True)


def _build(config, embedder, identity, dataset, fingerprint, name, chunks):
    client = _client(config.index_dir)
    collection = client.get_or_create_collection(
        name, embedding_function=None,
        configuration={"hnsw": {"space": "cosine"}},
        metadata={"distance": "cosine", "fingerprint": fingerprint, "state": "building"},
    )
    if (collection.metadata.get("fingerprint") != fingerprint
            or collection.metadata.get("distance") != "cosine"
            or collection.configuration["hnsw"]["space"] != "cosine"):
        raise RetrievalIndexError("Coleção incompatível; use outro diretório de índice")
    manifest_path = config.index_dir / "manifests" / (name + ".json")
    manifest = {"schema": 1, "state": "building", "collection": name,
                "fingerprint": fingerprint, **identity, "dataset": dataset,
                "chunk_count": len(chunks), "document_count": len({c.document_id for c in chunks}),
                "included_documents": sorted(
                    [{"document_id": document_id, "source": source} for document_id, source in
                     {c.document_id: c.source for c in chunks}.items()], key=lambda item: item["document_id"]),
                "chroma_configuration": collection.configuration,
                "dependencies": dependency_versions(),
                "ollama_version": getattr(embedder, "server_version", lambda: None)(),
                "updated_at": datetime.now(timezone.utc).isoformat()}
    collection.modify(metadata={**collection.metadata, "state": "building"})
    _atomic_json(manifest_path, manifest)
    embedded = 0
    print(f"index: {name}, {len(chunks)} trechos", file=sys.stderr, flush=True)
    for start in range(0, len(chunks), config.batch_size):
        batch = chunks[start:start + config.batch_size]
        existing = set(collection.get(ids=[c.id for c in batch], include=[]) ["ids"])
        pending = [c for c in batch if c.id not in existing]
        if not pending:
            continue
        vectors = embedder.documents(pending)
        collection.upsert(
            ids=[c.id for c in pending], documents=[c.text for c in pending], embeddings=vectors,
            metadatas=[{"document_id": c.document_id, "source": c.source, "title": c.title,
                        "metadata_json": json.dumps(c.metadata, ensure_ascii=False)} for c in pending],
        )
        embedded += len(pending)
        if start % (config.batch_size * 10) == 0 or start + config.batch_size >= len(chunks):
            print(f"index: {min(start + config.batch_size, len(chunks))}/{len(chunks)}, "
                  f"{embedded} embeddings novos", file=sys.stderr, flush=True)
    if collection.count() != len(chunks):
        raise RetrievalIndexError("Contagem incompatível; a coleção permaneceu incompleta")
    if embedder.model_digest() != identity["model_digest"]:
        raise RetrievalIndexError("O modelo mudou durante a indexação; repita com o modelo estável")
    collection.modify(metadata={**collection.metadata, "state": "ready"})
    manifest.update(state="ready", updated_at=datetime.now(timezone.utc).isoformat())
    _atomic_json(manifest_path, manifest)
    return {**manifest, "embedded_this_run": embedded, "reused_chunks": len(chunks) - embedded,
            "manifest_path": str(manifest_path)}


class Retriever:
    """Open an existing compatible index. Each result is a JSON-compatible dict."""

    def __init__(self, config=None, embedder=None):
        self.config = config or RetrievalConfig.from_env()
        self.embedder = embedder or OllamaEmbeddings(self.config)
        if not (self.config.index_dir / "chroma.sqlite3").is_file():
            raise RetrievalIndexError("Índice ausente. Execute python -m app.retrieval build")
        identity, _, fingerprint, name = _plan(self.config, self.embedder)
        path = self.config.index_dir / "manifests" / (name + ".json")
        if not path.is_file():
            raise RetrievalIndexError("Índice ausente ou incompatível com corpus/modelo/chunking. "
                                      "Execute build para esta configuração")
        try:
            self.manifest = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(self.manifest, dict):
                raise ValueError("manifesto deve ser objeto JSON")
        except (ValueError, OSError) as exc:
            raise RetrievalIndexError(f"Manifesto de índice inválido: {exc}") from exc
        if (self.manifest.get("state") != "ready" or self.manifest.get("fingerprint") != fingerprint
                or any(self.manifest.get(k) != v for k, v in identity.items())):
            raise RetrievalIndexError("Índice incompleto ou incompatível; execute build")
        self.client = _client(self.config.index_dir)
        try:
            self.collection = self.client.get_collection(name, embedding_function=None)
        except Exception as exc:
            raise RetrievalIndexError(f"Coleção {name} ausente ou ilegível: {exc}") from exc
        if (self.collection.metadata.get("fingerprint") != fingerprint
                or self.collection.metadata.get("state") != "ready"
                or self.collection.metadata.get("distance") != "cosine"
                or self.collection.configuration["hnsw"]["space"] != "cosine"
                or self.collection.count() != self.manifest.get("chunk_count")):
            raise RetrievalIndexError("Coleção incompleta ou incompatível; execute build")
        self.collection_name = name

    def retrieve(self, question, top_k=None):
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question deve ser texto não vazio")
        k = self.config.top_k if top_k is None else top_k
        if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
            raise ValueError("top_k deve ser inteiro positivo")
        vector = self.embedder.query(question.strip())
        if self.embedder.model_digest() != self.manifest["model_digest"]:
            raise RetrievalIndexError("O modelo do Ollama mudou desde a abertura do índice; reabra com a configuração compatível")
        result = self.collection.query(
            query_embeddings=[vector],
            n_results=min(k, self.collection.count()), include=["documents", "metadatas", "distances"],
        )
        rows = []
        for chunk_id, text, metadata, distance in zip(
                result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]):
            if not math.isfinite(distance):
                raise RetrievalIndexError("Chroma retornou distância não finita")
            rows.append({"chunk_id": chunk_id, "document_id": metadata["document_id"],
                         "text": text, "source": metadata["source"], "title": metadata["title"],
                         "metadata": json.loads(metadata["metadata_json"]),
                         "distance": distance, "distance_metric": "cosine"})
        return rows
