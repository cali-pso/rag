import math
import re
from collections import Counter
from functools import lru_cache
from typing import Dict, List, Tuple
import numpy as np
from src.models import MinimalSource

# Matches on the ORIGINAL-case text, before lowering: case information
# is what lets camelCase boundaries be found at all, and it's gone
# once everything is lowercased.
_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9_]+")
_CAMEL_BOUNDARY_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


@lru_cache(maxsize=None)
def _expand_identifier(raw: str) -> Tuple[str, ...]:
    """Returns the lowered identifier followed, for a compound one, by
    its lowered snake_case/camelCase parts.

    Memoized: code repeats the same identifiers constantly (~55k distinct
    terms for millions of occurrences in the vLLM corpus), and this split
    used to be redone for every single occurrence.
    """
    parts: List[str] = []
    for snake_part in raw.split("_"):
        parts.extend(p for p in _CAMEL_BOUNDARY_RE.split(snake_part) if p)
    if len(parts) > 1:
        return (raw.lower(), *(p.lower() for p in parts))
    return (raw.lower(),)


class BM25Retriever:
    """Okapi BM25 over the indexed chunks, backed by an inverted index.

    fit() stores, for every term, the chunks containing it together with
    the term's full BM25 contribution to each of them: nothing in that
    contribution depends on the query, so a search only has to add up
    precomputed numbers for the chunks sharing a term with the query,
    instead of scoring every chunk of the corpus.

    The postings of every term live in two flat arrays grouped by term:
    those of term t are doc_ids[offsets[t]:offsets[t + 1]] (ascending
    chunk order), and impacts holds the matching BM25 contributions.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        """
        Parameters:
            k1 (float): Term-frequency saturation.
            b (float): Strength of the document-length normalization.
        """
        self.k1 = k1
        self.b = b
        self.sources: List[MinimalSource] = []
        self.avgdl = 0.0
        self.vocabulary: Dict[str, int] = {}
        self.offsets = np.zeros(1, dtype=np.int64)
        self.doc_ids = np.zeros(0, dtype=np.int64)
        self.impacts = np.zeros(0, dtype=np.float64)

    def tokenize(self, text: str) -> List[str]:
        """Extracts identifier-like tokens, keeping digits and
        underscores as part of the same token so identifiers like
        `trust_remote_code` or constants like `FP8_MIN` survive as
        single terms. Each compound identifier is additionally indexed
        by its snake_case/camelCase parts (Gemma2DecoderLayer ->
        gemma2decoderlayer, gemma2, decoder, layer), so a question
        referencing part of a name without quoting it verbatim still
        gets partial credit."""
        tokens: List[str] = []
        for raw in _IDENTIFIER_RE.findall(text):
            tokens.extend(_expand_identifier(raw))
        return tokens

    def fit(self, sources: List[MinimalSource]) -> None:
        """Builds the inverted index and precomputes every term's BM25
        contribution to every chunk containing it.

        Args:
            sources: The chunks to search; replaces any previous corpus.
        """
        self.sources = list(sources)
        self.vocabulary = {}
        num_docs = len(self.sources)

        # One (term id, chunk index, term frequency) triple per distinct
        # term of each chunk, in flat lists: a handful of big arrays
        # instead of one small array pair per vocabulary term. Each
        # chunk's triples are appended in bulk, and chunk indices are
        # only expanded at the end (np.repeat), to keep Python-level work
        # per posting to a minimum.
        posting_terms: List[int] = []
        posting_tfs: List[int] = []
        distinct_terms = np.zeros(num_docs, dtype=np.int64)
        doc_lengths = np.zeros(num_docs, dtype=np.float64)
        vocabulary = self.vocabulary
        for i, source in enumerate(self.sources):
            tokens = self.tokenize(source.file_path + "\n" + source.content)
            counts = Counter(tokens)
            doc_lengths[i] = len(tokens)
            distinct_terms[i] = len(counts)
            posting_terms.extend(
                [vocabulary.setdefault(t, len(vocabulary)) for t in counts]
            )
            posting_tfs.extend(counts.values())

        terms = np.array(posting_terms, dtype=np.int64)
        docs = np.repeat(np.arange(num_docs, dtype=np.int64), distinct_terms)
        tfs = np.array(posting_tfs, dtype=np.float64)
        dfs = np.bincount(terms, minlength=len(self.vocabulary))

        self.avgdl = float(doc_lengths.sum()) / num_docs if num_docs else 0.0
        # A corpus without a single token has no postings to score, but
        # the division below must still be defined.
        safe_avgdl = self.avgdl if self.avgdl > 0 else 1.0
        # Same expressions, evaluated in the same order, as the textbook
        # per-document loop, so every score is bit-for-bit identical
        # (math.log rather than np.log for the same reason).
        norm_factors = 1 - self.b + self.b * (doc_lengths / safe_avgdl)
        idf = np.array(
            [
                math.log(((num_docs - df + 0.5) / (df + 0.5)) + 1.0)
                for df in dfs.tolist()
            ],
            dtype=np.float64,
        )
        numerator = tfs * (self.k1 + 1)
        denominator = tfs + self.k1 * norm_factors[docs]
        impacts = idf[terms] * (numerator / denominator)

        # Group the postings by term; the stable sort keeps each term's
        # chunks in ascending corpus order.
        order = np.argsort(terms, kind="stable")
        self.doc_ids = docs[order]
        self.impacts = impacts[order]
        self.offsets = np.concatenate(([0], np.cumsum(dfs))).astype(np.int64)

    def search(self, query: str, top_k: int = 5) -> List[MinimalSource]:
        """Returns the top_k chunks with the highest BM25 score.

        Only chunks sharing at least one term with the query are scored;
        ties keep corpus order.

        Args:
            query: The natural-language question to search for.
            top_k: Maximum number of chunks to return.

        Returns:
            Matching chunks, best first; chunks scoring 0 are left out.
        """
        if top_k <= 0 or not self.sources:
            return []

        scores = np.zeros(len(self.sources), dtype=np.float64)
        # Repeated query tokens count once per occurrence, as in the
        # classic formulation. A chunk appears at most once per postings
        # list, so the fancy-indexed += never drops an addition.
        for token in self.tokenize(query):
            term_id = self.vocabulary.get(token)
            if term_id is not None:
                start, end = self.offsets[term_id], self.offsets[term_id + 1]
                scores[self.doc_ids[start:end]] += self.impacts[start:end]

        candidates = np.flatnonzero(scores > 0.0)
        # Sort by score descending, then by corpus position ascending.
        order = np.lexsort((candidates, -scores[candidates]))[:top_k]
        return [self.sources[i] for i in candidates[order]]
