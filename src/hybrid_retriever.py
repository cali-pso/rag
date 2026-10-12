from typing import Dict, List
from src.bm25retriever import BM25Retriever
from src.vector_indexer import VectorIndexer
from src.models import MinimalSource

# Reciprocal Rank Fusion constant. The usual ~60 default comes from
# web-scale search literature; on a corpus this size it flattens rank
# differences almost to nothing (1/(1+60) vs 1/(5+60) differ by ~6%), so
# a strong rank-1 hit barely outweighs a mediocre one. A smaller constant
# keeps top ranks meaningfully more influential than the rest of the pool.
RRF_CONSTANT = 10
# BM25 is the stronger, better-measured signal on this corpus (see the
# evaluate command); semantic search contributes at half strength so it
# can still rescue lexical misses without outvoting a correct BM25 hit.
LEXICAL_WEIGHT = 1.0
SEMANTIC_WEIGHT = 0.5
# Each retriever is asked for more than the final k before fusing, so a
# chunk absent from one retriever's top-k still has room to surface from
# the other's wider candidate pool.
FETCH_MULTIPLIER = 4
MIN_FETCH = 20


class HybridRetriever:
    """Combines lexical (BM25) and semantic (FAISS) rankings into a single
    result list via weighted Reciprocal Rank Fusion.

    BM25 scores and cosine similarities live on incomparable scales, so
    fusion is done on rank position rather than raw scores; the two
    sides are weighted rather than summed evenly since they aren't
    equally reliable on this corpus.
    """

    def __init__(
        self,
        bm25: BM25Retriever,
        vector_indexer: VectorIndexer,
        rrf_constant: int = RRF_CONSTANT,
        lexical_weight: float = LEXICAL_WEIGHT,
        semantic_weight: float = SEMANTIC_WEIGHT,
    ) -> None:
        """
        Parameters:
            bm25 (BM25Retriever): An already-fitted lexical retriever.
            vector_indexer (VectorIndexer): An already-loaded semantic index.
            rrf_constant (int): The RRF rank-damping constant.
            lexical_weight (float): Contribution multiplier for BM25 ranks.
            semantic_weight (float): Contribution multiplier for semantic ranks.
        """
        self.bm25 = bm25
        self.vector_indexer = vector_indexer
        self.rrf_constant = rrf_constant
        self.lexical_weight = lexical_weight
        self.semantic_weight = semantic_weight

    @staticmethod
    def _fetch_count(k: int) -> int:
        """How many candidates to pull from each retriever before fusing."""
        return max(k * FETCH_MULTIPLIER, MIN_FETCH)

    def _fuse(
        self,
        lexical_results: List[MinimalSource],
        semantic_results: List[MinimalSource],
        k: int,
    ) -> List[MinimalSource]:
        """Merges the two ranked source lists into one, by summing each
        source's weighted reciprocal rank across whichever list(s) it
        appears in."""
        scores: Dict[int, float] = {}
        sources_by_id: Dict[int, MinimalSource] = {}

        for weight, ranked_list in (
            (self.lexical_weight, lexical_results),
            (self.semantic_weight, semantic_results),
        ):
            for rank, source in enumerate(ranked_list, start=1):
                source_id = source.faiss_custom_id
                if source_id is None:
                    continue
                scores[source_id] = scores.get(source_id, 0.0) + weight / (
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
        if k <= 0 or not query.strip():
            return []

        fetch_n = self._fetch_count(k)
        lexical_results = self.bm25.search(query, fetch_n)
        semantic_results = self.vector_indexer.search(query, fetch_n)
        return self._fuse(lexical_results, semantic_results, k)

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
        results: List[List[MinimalSource]] = [[] for _ in queries]
        if k <= 0:
            return results

        # Blank questions keep their empty result: any vector is "close"
        # to the embedding of nothing, so searching them returns noise.
        positions = [i for i, query in enumerate(queries) if query.strip()]
        if not positions:
            return results

        fetch_n = self._fetch_count(k)
        semantic_results = self.vector_indexer.search_batch(
            [queries[i] for i in positions], fetch_n
        )
        for i, semantic in zip(positions, semantic_results):
            lexical = self.bm25.search(queries[i], fetch_n)
            results[i] = self._fuse(lexical, semantic, k)
        return results
