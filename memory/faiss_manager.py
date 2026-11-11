"""FAISS long-term memory for the FRAME pipeline.

Vectorizes the Stage 2 outputs — the Reflector's feedback/reflections and the
generated training instances — into a FAISS index so that, at inference time,
the Filter agent can retrieve historical support for a proposed research topic.

Embedding backend: by default a lightweight sentence-transformers model
(``config.EMBEDDING_MODEL``).  The embedder is pluggable — pass any callable
``embed(text) -> list[float]`` to :class:`MemoryManager`.

Heavy deps (``faiss`` / ``numpy`` / ``sentence-transformers``) are imported
lazily so importing this module never fails; a clear error is raised only when
you actually build or query the index.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import config
from data.storage import load_stage2_results


def _import_faiss():
    try:
        import faiss  # type: ignore
        return faiss
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "faiss-cpu is not installed. Run: pip install faiss-cpu"
        ) from exc


def _import_numpy():
    try:
        import numpy as np  # type: ignore
        return np
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("numpy is not installed. Run: pip install numpy") from exc


class SentenceTransformerEmbedder:
    """Lazy sentence-transformers wrapper exposing ``embed(text) -> list[float]``."""

    def __init__(self, model_name: Optional[str] = None) -> None:
        self.model_name = model_name or config.EMBEDDING_MODEL
        self._model = None

    def embed(self, text: str) -> List[float]:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
            except ImportError as exc:  # pragma: no cover
                raise RuntimeError(
                    "sentence-transformers is not installed. "
                    "Run: pip install sentence-transformers"
                ) from exc
            self._model = SentenceTransformer(self.model_name)
        return self._model.encode([text], normalize_embeddings=True)[0].tolist()


class MemoryManager:
    """Build, persist, and query the FAISS long-term memory."""

    def __init__(
        self,
        index_path: Optional[Union[str, Path]] = None,
        metadata_path: Optional[Union[str, Path]] = None,
        embedder: Optional[Callable[[str], List[float]]] = None,
    ) -> None:
        self.index_path = Path(index_path) if index_path else Path(config.FAISS_INDEX_PATH)
        self.metadata_path = (
            Path(metadata_path) if metadata_path else Path(config.FAISS_METADATA_PATH)
        )
        self._embedder = embedder or SentenceTransformerEmbedder()
        self._index = None
        self._metadata: List[Dict[str, Any]] = []

    # -- embedding --------------------------------------------------------
    def _embed(self, texts: List[str]):
        np = _import_numpy()
        vectors = np.array([self._embedder.embed(t) for t in texts], dtype="float32")
        # L2-normalize so inner-product search == cosine similarity.
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return np.ascontiguousarray(vectors / norms, dtype="float32")

    # -- chunk extraction -------------------------------------------------
    @staticmethod
    def _extract_chunks(stage2: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Turn Stage 2 reflections + instances into indexable text chunks."""
        chunks: List[Dict[str, Any]] = []
        for i, ref in enumerate(stage2.get("reflections", [])):
            feedback = str(ref.get("feedback", "")).strip()
            if feedback:
                chunks.append(
                    {
                        "id": f"reflection_{i}",
                        "source": "reflection",
                        "text": feedback,
                        "round": ref.get("round"),
                        "reason": str(ref.get("reason", "")).strip(),
                    }
                )
        for i, inst in enumerate(stage2.get("instances", [])):
            text = " / ".join(
                str(inst.get(k, "")).strip()
                for k in ("instruction", "input", "output")
                if str(inst.get(k, "")).strip()
            )
            if text:
                chunks.append(
                    {"id": f"instance_{i}", "source": "instance", "text": text}
                )
        return chunks

    # -- build / update ---------------------------------------------------
    def build_or_update_index(
        self,
        stage2_path: Optional[Union[str, Path]] = None,
    ) -> int:
        """Rebuild the FAISS index from ``stage2_results.json``.

        Returns the number of chunks indexed.  Creates the index file if it
        does not exist yet.
        """
        faiss = _import_faiss()

        stage2 = load_stage2_results(stage2_path)
        chunks = self._extract_chunks(stage2)
        if not chunks:
            print("[memory] no chunks found in stage2_results.json; index left empty.")
            self._index = None
            self._metadata = []
            return 0

        vectors = self._embed([c["text"] for c in chunks])
        dim = int(vectors.shape[1])

        self._index = faiss.IndexFlatIP(dim)
        self._index.add(vectors)
        self._metadata = chunks
        self._save(faiss)
        return len(chunks)

    # -- query ------------------------------------------------------------
    def query_memory(self, query_text: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Return the ``top_k`` most relevant historical chunks (with scores)."""
        faiss = _import_faiss()
        self._load_if_exists(faiss)

        if self._index is None or self._index.ntotal == 0:
            return []

        top_k = min(top_k, self._index.ntotal)
        query_vec = self._embed([query_text])
        scores, ids = self._index.search(query_vec, top_k)

        results: List[Dict[str, Any]] = []
        for score, idx in zip(scores[0], ids[0]):
            idx = int(idx)
            if idx == -1:
                continue
            meta = dict(self._metadata[idx])
            meta["score"] = float(score)
            results.append(meta)
        return results

    # -- persistence ------------------------------------------------------
    def _save(self, faiss) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(self.index_path))
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with self.metadata_path.open("w", encoding="utf-8") as fh:
            json.dump(self._metadata, fh, ensure_ascii=False, indent=2)

    def _load_if_exists(self, faiss) -> None:
        if self._index is not None:
            return
        if self.index_path.exists() and self.metadata_path.exists():
            self._index = faiss.read_index(str(self.index_path))
            self._metadata = json.loads(
                self.metadata_path.read_text(encoding="utf-8")
            )
        else:
            self._index = None
            self._metadata = []
