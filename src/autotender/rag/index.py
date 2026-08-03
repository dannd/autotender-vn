"""FAISS index wrapper cho bi-encoder embeddings (Tier 1/2 của M4 Retrieval, Mục 6/M4).

`IndexFlatIP` — dataset nhỏ (vài trăm-vài nghìn chunk), không cần HNSW.
Import `faiss`/`numpy` được trì hoãn để module này import được kể cả khi
chưa cài `faiss-cpu` (Tier 3 vẫn hoạt động bình thường không cần file này).
"""

from __future__ import annotations

from pathlib import Path


class FaissChunkIndex:
    def __init__(self, dim: int):
        import faiss

        self._faiss = faiss
        self.index = faiss.IndexFlatIP(dim)
        self.dim = dim

    def add(self, embeddings) -> None:
        import numpy as np

        vecs = np.asarray(embeddings, dtype="float32")
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        vecs = vecs / norms
        self.index.add(vecs)

    def search(self, query_embedding, top_k: int) -> tuple[list[int], list[float]]:
        import numpy as np

        vec = np.asarray([query_embedding], dtype="float32")
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        scores, indices = self.index.search(vec, top_k)
        return indices[0].tolist(), scores[0].tolist()

    def save(self, path: str | Path) -> None:
        self._faiss.write_index(self.index, str(path))

    @classmethod
    def load(cls, path: str | Path, dim: int) -> "FaissChunkIndex":
        import faiss

        obj = cls.__new__(cls)
        obj._faiss = faiss
        obj.dim = dim
        obj.index = faiss.read_index(str(path))
        return obj
