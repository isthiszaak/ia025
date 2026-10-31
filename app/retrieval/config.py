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
    # The constructor retains the original Python API's character defaults.
    # from_env() and the CLI default to the planned token configuration.
    chunk_unit: str = "characters"
    tokenizer_id: str = "onnx-community/embeddinggemma-300m-ONNX"
    tokenizer_revision: str = "8cd82455fa7fae53af154775fd1494f5577ef4b0"
    embedding_max_tokens: int = 2048

    def __post_init__(self):
        for name in ("dimension", "chunk_size", "top_k", "batch_size", "embedding_max_tokens"):
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
        if self.chunk_unit not in {"characters", "tokens"}:
            raise ValueError("chunk_unit deve ser characters ou tokens")
        if self.profile == "embeddinggemma" and self.embedding_max_tokens > 2048:
            raise ValueError("EmbeddingGemma aceita no máximo 2048 tokens de entrada")
        if self.chunk_unit == "tokens":
            import re
            if not self.tokenizer_id.strip() or not re.fullmatch(r"[0-9a-f]{40}", self.tokenizer_revision):
                raise ValueError("Tokenizer exige identificação e revisão Git SHA de 40 caracteres")
        parsed = urlparse(self.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.path not in {"", "/"}:
            raise ValueError("RETRIEVAL_OLLAMA_BASE_URL deve ser a raiz HTTP do Ollama, sem /v1")

    @classmethod
    def from_env(cls):
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env", override=False)
        selection = os.getenv("RETRIEVAL_CORPUS_SELECTION", str(ROOT / "retrieval_data/baseline_irpf.json"))
        # Old .env files named character sizes without a unit. Never silently
        # reinterpret an existing configuration as token counts.
        legacy = "RETRIEVAL_CHUNK_SIZE" in os.environ or "RETRIEVAL_CHUNK_OVERLAP" in os.environ
        unit = os.getenv("RETRIEVAL_CHUNK_UNIT", "characters" if legacy else "tokens")
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
            chunk_unit=unit,
            chunk_size=int(os.getenv("RETRIEVAL_CHUNK_SIZE", "512" if unit == "tokens" else "1200")),
            chunk_overlap=int(os.getenv("RETRIEVAL_CHUNK_OVERLAP", "50" if unit == "tokens" else "200")),
            tokenizer_id=os.getenv("RETRIEVAL_TOKENIZER_ID", cls.tokenizer_id),
            tokenizer_revision=os.getenv("RETRIEVAL_TOKENIZER_REVISION", cls.tokenizer_revision),
            embedding_max_tokens=int(os.getenv("RETRIEVAL_EMBEDDING_MAX_TOKENS", "2048")),
            top_k=int(os.getenv("RETRIEVAL_TOP_K", "5")),
            batch_size=int(os.getenv("RETRIEVAL_BATCH_SIZE", "32")),
            timeout_seconds=float(os.getenv("RETRIEVAL_TIMEOUT_SECONDS", "180")),
        )

    def index_parameters(self):
        from .chunking import CHUNKING_VERSION
        # Operational settings and top-k do not change the indexed vectors.
        parameters = {"model": self.model, "dimension": self.dimension, "profile": self.profile,
                "chunk_size": self.chunk_size, "chunk_overlap": self.chunk_overlap,
                "chunk_unit": "characters", "chunk_algorithm": CHUNKING_VERSION,
                "embedding_processing": "prefix-v1", "distance": "cosine"}
        if self.chunk_unit == "tokens":
            from .chunking import TOKEN_CHUNKING_VERSION
            parameters.update(chunk_unit="tokens", chunk_algorithm=TOKEN_CHUNKING_VERSION,
                              tokenizer=self.load_tokenizer().identity(),
                              embedding_max_tokens=self.embedding_max_tokens)
        return parameters

    def load_tokenizer(self):
        from .tokenizer import load_tokenizer
        return load_tokenizer(self.data_dir / "tokenizer", self.tokenizer_id, self.tokenizer_revision)
