"""Configuration isolated from generation and evaluation settings."""

from dataclasses import dataclass
import math
import os
from pathlib import Path
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class RetrievalConfig:
    data_dir: Path = ROOT / "data/retrieval"
    index_dir: Path = ROOT / "data/retrieval/chroma"
    corpus_selection: Path | None = ROOT / "retrieval_data/baseline_irpf.json"
    base_url: str = "http://localhost:11435"
    model: str = "embeddinggemma:300m"
    dimension: int = 768
    profile: str = "embeddinggemma"
    chunk_size: int = 1200
    chunk_overlap: int = 200
    top_k: int = 5
    batch_size: int = 32
    timeout_seconds: float = 180

    def __post_init__(self):
        for name in ("dimension", "chunk_size", "top_k", "batch_size"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} deve ser inteiro positivo")
        if (isinstance(self.chunk_overlap, bool) or not isinstance(self.chunk_overlap, int)
                or not 0 <= self.chunk_overlap < self.chunk_size):
            raise ValueError("chunk_overlap deve estar entre 0 e chunk_size - 1")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds deve ser finito e positivo")
        if not self.model.strip() or self.profile not in {"embeddinggemma", "plain"}:
            raise ValueError("Modelo vazio ou profile desconhecido (embeddinggemma/plain)")
        parsed = urlparse(self.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.path not in {"", "/"}:
            raise ValueError("RETRIEVAL_OLLAMA_BASE_URL deve ser a raiz HTTP do Ollama, sem /v1")

    @classmethod
    def from_env(cls):
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env", override=False)
        selection = os.getenv("RETRIEVAL_CORPUS_SELECTION", str(ROOT / "retrieval_data/baseline_irpf.json"))
        def local_path(value):
            result = Path(value)
            return result if result.is_absolute() else ROOT / result
        return cls(
            data_dir=local_path(os.getenv("RETRIEVAL_DATA_DIR", str(ROOT / "data/retrieval"))),
            index_dir=local_path(os.getenv("RETRIEVAL_INDEX_DIR", str(ROOT / "data/retrieval/chroma"))),
            corpus_selection=None if selection.lower() == "all" else local_path(selection),
            base_url=os.getenv("RETRIEVAL_OLLAMA_BASE_URL", "http://localhost:11435"),
            model=os.getenv("RETRIEVAL_EMBEDDING_MODEL", "embeddinggemma:300m"),
            dimension=int(os.getenv("RETRIEVAL_EMBEDDING_DIMENSION", "768")),
            profile=os.getenv("RETRIEVAL_EMBEDDING_PROFILE", "embeddinggemma"),
            chunk_size=int(os.getenv("RETRIEVAL_CHUNK_SIZE", "1200")),
            chunk_overlap=int(os.getenv("RETRIEVAL_CHUNK_OVERLAP", "200")),
            top_k=int(os.getenv("RETRIEVAL_TOP_K", "5")),
            batch_size=int(os.getenv("RETRIEVAL_BATCH_SIZE", "32")),
            timeout_seconds=float(os.getenv("RETRIEVAL_TIMEOUT_SECONDS", "180")),
        )

    def index_parameters(self):
        from .chunking import CHUNKING_VERSION
        # Operational settings and top-k do not change the indexed vectors.
        return {"model": self.model, "dimension": self.dimension, "profile": self.profile,
                "chunk_size": self.chunk_size, "chunk_overlap": self.chunk_overlap,
                "chunk_unit": "characters", "chunk_algorithm": CHUNKING_VERSION,
                "embedding_processing": "prefix-v1", "distance": "cosine"}
