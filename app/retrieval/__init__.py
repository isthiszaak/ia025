"""Public API: Retriever(config).retrieve(question, top_k=None)."""

from .config import RetrievalConfig


def __getattr__(name):
    if name in {"Retriever", "RetrievalIndexError", "build_index"}:
        from . import index
        return getattr(index, name)
    raise AttributeError(name)
