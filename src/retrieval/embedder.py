"""Embedding wrapper with graceful degradation.

Default mode is `local` (sentence-transformers). If that package is missing,
or mode is `off`, the embedder reports `available == False` and the retriever
falls back to BM25-only. Real embeddings are an upgrade, not a requirement.
"""
from __future__ import annotations

from ..config import settings


class Embedder:
    def __init__(self):
        self.mode = settings.embeddings_mode
        self._model = None
        if self.mode == "local":
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(settings.embeddings_model)
            except Exception:
                self._model = None

    @property
    def available(self) -> bool:
        if self.mode == "off":
            return False
        if self.mode == "local":
            return self._model is not None
        if self.mode == "openai":
            return bool(settings.openai_api_key)
        return False

    def embed(self, texts: list[str]):
        if self.mode == "local" and self._model is not None:
            return self._model.encode(texts, convert_to_numpy=True).tolist()
        if self.mode == "openai" and settings.openai_api_key:
            from langchain_openai import OpenAIEmbeddings

            e = OpenAIEmbeddings(model=settings.openai_embeddings_model, api_key=settings.openai_api_key)
            return e.embed_documents(texts)
        return None
