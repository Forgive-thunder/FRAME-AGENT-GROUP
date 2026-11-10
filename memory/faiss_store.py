"""Minimal FAISS vector-knowledge-base placeholder.

Only the public interface is defined.  The methods are intentionally empty
stubs — the user asked for ``add_memory`` / ``search_memory`` shells — so the
scaffold documents *where* retrieval plugs in without depending on a working
index yet.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


class FAISSVectorStore:
    """Thin wrapper around a future ``faiss`` index.

    ``dimension`` and ``index_path`` mirror the config knobs in ``config.py``;
    they are recorded now so the real implementation has a stable constructor.
    """

    def __init__(
        self,
        dimension: int = 768,
        index_path: Optional[str] = None,
    ) -> None:
        self.dimension = dimension
        self.index_path = index_path
        self._index = None  # TODO: faiss.IndexFlatL2(dimension)
        self._metadata: Dict[str, Dict[str, Any]] = {}

    def add_memory(
        self,
        key: str,
        vector: List[float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Store a vector under ``key``.

        TODO: normalize the vector, insert into ``self._index`` (an
        ``faiss.IndexFlatL2``), and persist ``metadata`` keyed by index id.
        """
        raise NotImplementedError("FAISSVectorStore.add_memory is a stub.")

    def search_memory(self, query_vector: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        """Return the ``top_k`` nearest neighbours to ``query_vector``.

        TODO: ``self._index.search(...)``, then map returned ids back to the
        metadata recorded by :meth:`add_memory`.
        """
        raise NotImplementedError("FAISSVectorStore.search_memory is a stub.")
