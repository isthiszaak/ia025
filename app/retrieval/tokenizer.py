"""Pinned EmbeddingGemma tokenizer; no model weights or approximate counts.

Only tokenizer.json is downloaded from the ONNX Community conversion of
google/embeddinggemma-300m. Its revision and published SHA256 are pinned.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from importlib.metadata import version
import json
from pathlib import Path
import re
import tempfile
import urllib.error
import urllib.request


DEFAULT_TOKENIZER = "onnx-community/embeddinggemma-300m-ONNX"
DEFAULT_TOKENIZER_REVISION = "8cd82455fa7fae53af154775fd1494f5577ef4b0"
DEFAULT_TOKENIZER_SHA256 = "4dda02faaf32bc91031dc8c88457ac272b00c1016cc679757d1c441b248b9c47"


class TokenizerError(RuntimeError):
    pass


@lru_cache(maxsize=2)
def _parse_backend(raw: bytes):
    # Integrity is checked before every call. Reuse the expensive parsed
    # vocabulary only when the exact verified bytes are unchanged.
    from tokenizers import Tokenizer
    return Tokenizer.from_str(raw.decode("utf-8"))


class TextTokenizer:
    """Thin adapter around a real Hugging Face ``tokenizers.Tokenizer``.

    Encoding disables truncation/padding. Offsets refer to Python Unicode
    character positions in the original text, including non-ASCII characters.
    The backend's postprocessor supplies its declared special tokens when
    counting the complete embedding input.
    """

    def __init__(self, backend, identifier: str, revision: str, sha256: str):
        backend.no_truncation()
        backend.no_padding()
        self.backend = backend
        self._identity = {
            "id": identifier, "revision": revision, "sha256": sha256,
            "library": "tokenizers", "library_version": version("tokenizers"),
        }

    def identity(self) -> dict:
        return dict(self._identity)

    def encode(self, text: str, add_special_tokens: bool = False):
        return self.backend.encode(text, add_special_tokens=add_special_tokens)

    def count(self, text: str, add_special_tokens: bool = True) -> int:
        return len(self.encode(text, add_special_tokens=add_special_tokens).ids)


def _write_atomic(path: Path, content: bytes):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_tokenizer(
    cache_dir: Path,
    model_id: str = DEFAULT_TOKENIZER,
    revision: str = DEFAULT_TOKENIZER_REVISION,
) -> TextTokenizer:
    """Download once, verify every read, then load a pinned tokenizer JSON.

    Other explicitly pinned tokenizer revisions are supported, with their
    acquired SHA256 retained in a sidecar and in every index identity. The
    default artifact is additionally checked against its published LFS hash.
    """
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", model_id):
        raise ValueError("tokenizer_id must be a Hugging Face owner/model identifier")
    if not re.fullmatch(r"[a-fA-F0-9]{40}", revision):
        raise ValueError("tokenizer_revision must be a pinned 40-character Git SHA")
    revision = revision.lower()
    directory = Path(cache_dir) / model_id.replace("/", "--") / revision
    path = directory / "tokenizer.json"
    sidecar = directory / "source.json"
    url = f"https://huggingface.co/{model_id}/resolve/{revision}/tokenizer.json"
    directory.mkdir(parents=True, exist_ok=True)
    downloaded = not path.exists()
    if downloaded:
        request = urllib.request.Request(url, headers={"User-Agent": "ia025-data-retrieval/1"})
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                content = response.read(64 * 1024 * 1024 + 1)
            if len(content) > 64 * 1024 * 1024:
                raise TokenizerError("tokenizer.json exceeds the supported 64 MiB limit")
            json.loads(content)
            _write_atomic(path, content)
        except (OSError, ValueError, urllib.error.URLError) as exc:
            raise TokenizerError(f"Could not download pinned tokenizer from {url}: {exc}") from exc
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    official = (model_id, revision) == (DEFAULT_TOKENIZER, DEFAULT_TOKENIZER_REVISION)
    if official:
        if digest != DEFAULT_TOKENIZER_SHA256:
            raise TokenizerError(f"Pinned tokenizer SHA256 mismatch: {path}")
    identity = {"id": model_id, "revision": revision, "sha256": digest}
    provenance = {
        **identity, "url": url if downloaded or official else None,
        "acquisition": "download" if downloaded else "verified_official_cache" if official else "local_file",
    }
    if sidecar.exists():
        try:
            previous = json.loads(sidecar.read_text(encoding="utf-8"))
        except (ValueError, OSError) as exc:
            raise TokenizerError(f"Invalid tokenizer provenance: {sidecar}") from exc
        if not isinstance(previous, dict) or any(previous.get(key) != value for key, value in identity.items()):
            raise TokenizerError(f"Tokenizer cache provenance or SHA256 mismatch: {path}")
    try:
        backend = _parse_backend(raw)
    except Exception as exc:
        raise TokenizerError(f"Invalid tokenizer JSON: {path}: {exc}") from exc
    if not sidecar.exists():
        _write_atomic(sidecar, (json.dumps(provenance, sort_keys=True) + "\n").encode("utf-8"))
    return TextTokenizer(backend, model_id, revision, digest)
