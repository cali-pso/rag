from typing import Dict, List
from src.bm25retriever import BM25Retriever
from src.vector_indexer import VectorIndexer
from src.models import MinimalSource

# Reciprocal Rank Fusion constant. Standard literature default; tune via
# the evaluate command if recall on either dataset plateaus below target.
RRF_CONSTANT = 60
# Each retriever is asked for more than the final k before fusing, so a
# chunk absent from one retriever's top-k still has room to surface from
# the other's wider candidate pool.
FETCH_MULTIPLIER = 4
MIN_FETCH = 20


class HybridRetriever:
    """Combines lexical (BM25) and semantic (FAISS) rankings into a single
    result list via Reciprocal Rank Fusion.

    BM25 scores and cosine similarities live on incomparable scales, so
    fusion is done purely on rank position rather than raw scores.
    """

    def __init__(
        self,
        bm25: BM25Retriever,
        vector_indexer: VectorIndexer,
        rrf_constant: int = RRF_CONSTANT,
    ) -> None:
        """
        Parameters:
            bm25 (BM25Retriever): An already-fitted lexical retriever.
            vector_indexer (VectorIndexer): An already-loaded semantic index.
            rrf_constant (int): The RRF rank-damping constant.
        """
        self.bm25 = bm25
        self.vector_indexer = vector_indexer
        self.rrf_constant = rrf_constant

    @staticmethod
    def _fetch_count(k: int) -> int:
        """How many candidates to pull from each retriever before fusing."""
        return max(k * FETCH_MULTIPLIER, MIN_FETCH)

    def _fuse(
        self, ranked_lists: List[List[MinimalSource]], k: int
    ) -> List[MinimalSource]:
        """Merges several ranked source lists into one, by summing each
        source's reciprocal rank across every list it appears in."""
        scores: Dict[int, float] = {}
        sources_by_id: Dict[int, MinimalSource] = {}

        for ranked_list in ranked_lists:
            for rank, source in enumerate(ranked_list, start=1):
                source_id = source.faiss_custom_id
                if source_id is None:
                    continue
                scores[source_id] = scores.get(source_id, 0.0) + 1.0 / (
                    rank + self.rrf_constant
                )
                sources_by_id.setdefault(source_id, source)

        ranked_ids = sorted(
            scores, key=lambda source_id: scores[source_id], reverse=True
        )
        return [sources_by_id[source_id] for source_id in ranked_ids[:k]]

    def search(self, query: str, k: int = 10) -> List[MinimalSource]:
        """Returns the top-k sources for a single query, fused from both
        retrievers.

        Parameters:
            query (str): The natural-language question to search for.
            k (int): Maximum number of sources to return.
        """
        if k <= 0:
            return []

        fetch_n = self._fetch_count(k)
        lexical_results = self.bm25.search(query, fetch_n)
        semantic_results = self.vector_indexer.search(query, fetch_n)
        return self._fuse([lexical_results, semantic_results], k)

    def search_batch(
        self, queries: List[str], k: int = 10
    ) -> List[List[MinimalSource]]:
        """Returns the top-k fused sources for many queries at once.

        The semantic side is embedded and searched in one batched pass;
        the lexical side stays per-query (BM25 is cheap enough per call
        that batching it wouldn't meaningfully help).

        Parameters:
            queries (List[str]): The natural-language questions to search for.
            k (int): Maximum number of sources to return per query.
        """
        if not queries or k <= 0:
            return [[] for _ in queries]

        fetch_n = self._fetch_count(k)
        semantic_results = self.vector_indexer.search_batch(queries, fetch_n)

        results = []
        for query, semantic in zip(queries, semantic_results):
            lexical = self.bm25.search(query, fetch_n)
            results.append(self._fuse([lexical, semantic], k))
        return results
