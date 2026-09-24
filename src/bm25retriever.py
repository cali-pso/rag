import math
import re
from pydantic import BaseModel
from collections import Counter
from typing import List, Dict
from src.models import MinimalSource

# Matches on the ORIGINAL-case text, before lowering: case information
# is what lets camelCase boundaries be found at all, and it's gone
# once everything is lowercased.
_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9_]+")
_CAMEL_BOUNDARY_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


class BM25Retriever(BaseModel):
    k1: float = 1.5
    b: float = 0.75
    sources: List[MinimalSource] = []
    doc_lengths: List[int] = []
    avgdl: float = 0.0
    document_frequencies: Dict[str, int] = {}
    term_frequencies: List[Counter] = []
    idf_cache: Dict[str, float] = {}

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
            tokens.append(raw.lower())
            parts: List[str] = []
            for snake_part in raw.split("_"):
                parts.extend(
                    p for p in _CAMEL_BOUNDARY_RE.split(snake_part) if p
                )
            if len(parts) > 1:
                tokens.extend(p.lower() for p in parts)
        return tokens

    def fit(self, sources: List[MinimalSource]) -> None:
        """Precomputes lengths, frequencies and IDF for the corpus"""
        self.sources = sources
        self.document_frequencies = Counter()

        total_length = 0
        for source in self.sources:
            tokens = self.tokenize(source.content)
            tokens_length = len(tokens)
            self.doc_lengths.append(tokens_length)
            total_length += tokens_length

            term_freq = Counter(tokens)
            self.term_frequencies.append(term_freq)

            for unique_term in term_freq.keys():
                self.document_frequencies[unique_term] += 1

        num_docs = len(self.sources)
        if num_docs > 0:
            self.avgdl = total_length / num_docs

        for term, df in self.document_frequencies.items():
            idf_value = math.log(((num_docs - df + 0.5) / (df + 0.5)) + 1.0)
            self.idf_cache[term] = idf_value

    def search(self, query: str, top_k: int = 5) -> List[MinimalSource]:
        """Scores all documents against the query and returns the top_k"""
        query_tokens = self.tokenize(query)
        scores = [0.0] * len(self.sources)

        for i, source in enumerate(self.sources):
            doc_len = self.doc_lengths[i]
            term_freq = self.term_frequencies[i]

            norm_factor = 1 - self.b + self.b * (doc_len / self.avgdl)

            for token in query_tokens:
                if token not in self.document_frequencies:
                    continue

                tf = term_freq[token]
                if tf == 0:
                    continue

                idf = self.idf_cache[token]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * norm_factor

                scores[i] += idf * (numerator / denominator)

        scored_sources = list(zip(scores, self.sources))
        scored_sources.sort(key=lambda x: x[0], reverse=True)

        return [
            source for score, source in scored_sources[:top_k] if score > 0.0
        ]
