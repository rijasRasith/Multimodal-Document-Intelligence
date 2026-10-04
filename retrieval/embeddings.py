from __future__ import annotations

from typing import Optional

from sentence_transformers import SentenceTransformer


_embedding_model: Optional[SentenceTransformer] = None
_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"


def get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(_MODEL_NAME)
    return _embedding_model


def embed_text(text: str) -> list[float]:
    model = get_embedding_model()
    embedding = model.encode(text, normalize_embeddings=True, show_progress_bar=False)
    return embedding.tolist()


def embed_batch(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    model = get_embedding_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return [e.tolist() for e in embeddings]
