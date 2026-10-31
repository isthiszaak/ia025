"""Small Ollama /api/embed client. Never silently truncate model input."""

import json
import math
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class EmbeddingError(RuntimeError):
    pass


class OllamaEmbeddings:
    def __init__(self, config):
        self.config = config
        self._tokenizer = None

    def _request(self, endpoint, payload=None):
        url = self.config.base_url.rstrip("/") + endpoint
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = Request(url, data=body, headers={"Content-Type": "application/json"})
        try:
            with urlopen(request, timeout=self.config.timeout_seconds) as response:
                result = json.load(response)
            if not isinstance(result, dict):
                raise EmbeddingError(f"Ollama {endpoint}: resposta JSON deve ser objeto")
            return result
        except HTTPError as exc:
            detail = exc.read(2000).decode("utf-8", errors="replace")
            raise EmbeddingError(f"Ollama {endpoint}: HTTP {exc.code}: {detail}") from exc
        except (URLError, TimeoutError, OSError, ValueError) as exc:
            raise EmbeddingError(f"Falha ao acessar Ollama em {url}: {exc}") from exc

    def model_digest(self):
        result = self._request("/api/tags")
        name = self.config.model
        names = {name, name + ":latest"} if ":" not in name else {name}
        for model in result.get("models", []):
            if model.get("name") in names or model.get("model") in names:
                digest = model.get("digest")
                if isinstance(digest, str) and digest:
                    return digest
        raise EmbeddingError(f"Modelo {name!r} ausente. Execute ollama pull {name} na mesma instância.")

    def server_version(self):
        return self._request("/api/version").get("version", "unknown")

    def _embed(self, texts):
        if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Embeddings exigem textos não vazios")
        if self.config.chunk_unit == "tokens":
            if self._tokenizer is None:
                self._tokenizer = self.config.load_tokenizer()
            for text in texts:
                count = self._tokenizer.count(text, add_special_tokens=True)
                if count > self.config.embedding_max_tokens:
                    raise EmbeddingError(
                        f"Entrada com {count} tokens (prefixo/título/especiais incluídos) excede "
                        f"o limite {self.config.embedding_max_tokens}; reduza o trecho/título/pergunta")
        result = self._request("/api/embed", {
            "model": self.config.model, "input": texts, "truncate": False,
        })
        vectors = result.get("embeddings")
        if not isinstance(vectors, list) or len(vectors) != len(texts):
            raise EmbeddingError("Ollama retornou quantidade inesperada de embeddings")
        for vector in vectors:
            if (not isinstance(vector, list) or len(vector) != self.config.dimension
                    or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                           for v in vector) or sum(v * v for v in vector) <= 0):
                raise EmbeddingError(f"Embedding inválido; dimensão esperada {self.config.dimension}")
        return vectors

    def documents(self, chunks):
        if self.config.profile == "embeddinggemma":
            texts = [f"title: {c.title or 'none'} | text: {c.text}" for c in chunks]
        else:
            texts = [c.text for c in chunks]
        return self._embed(texts)

    def query(self, question):
        text = (f"task: search result | query: {question}"
                if self.config.profile == "embeddinggemma" else question)
        return self._embed([text])[0]
