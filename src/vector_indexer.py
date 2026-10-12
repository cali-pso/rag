import os
import json
from typing import Dict, List, Set
import numpy as np
import openvino as ov
import faiss
from sentence_transformers import SentenceTransformer
from pydantic import ValidationError
from src.errors import CorruptedIndex
from src.models import MinimalSource

MODEL_NAME = "./openvino/openvino_weight_only_int8_quantized"
BATCH_SIZE = 16
# all-MiniLM-L6-v2 (and this quantized export of it) was fine-tuned with a
# 256-token max sequence length; beyond that, embeddings degrade even though
# the architecture's position embeddings go up to 512. 256 tokens covers a
# 2000-character chunk far better than the previous 128 without running the
# model outside the regime it was trained on.
MAX_SEQ_LENGTH = 256
FAISS_INDEX = "data/processed/faiss_index.bin"
FAISS_SOURCES = "data/processed/faiss_sources.json"


class VectorIndexer:
    def __init__(self, model_name: str = MODEL_NAME) -> None:
        """
        Initializes a custom model and the underlying FAISS index.

        Parameters:
            model_name (str): HuggingFace model identifier for sentence embeddings.
        """
        self.model = SentenceTransformer(
            MODEL_NAME,
            backend="openvino",
            model_kwargs={
                "device": "CPU",
                "ov_config": {
                    "INFERENCE_NUM_THREADS": str(os.cpu_count())
                    if os.cpu_count() is not None
                    else "4",
                    "NUM_STREAMS": "1",
                    "PERFORMANCE_HINT": "THROUGHPUT",
                    "ENABLE_CPU_PINNING": "YES",
                },
            },
        )
        self.model.max_seq_length = MAX_SEQ_LENGTH
        # Fix the batch dimension (sentence-transformers pads a short final
        # batch to match) but leave the sequence dimension bounded-dynamic:
        # a batch of short chunks is tokenized and run at its own actual
        # length instead of always paying for MAX_SEQ_LENGTH, while a batch
        # containing full-size chunks can still use up to the full ceiling.
        ov_model = self.model[0].auto_model.model
        # openvino's .pyi stub doesn't declare these (they only exist on the
        # compiled _pyopenvino extension re-exported at runtime).
        seq_dim = ov.Dimension(1, MAX_SEQ_LENGTH)  # type: ignore[attr-defined]
        ov_model.reshape({
            "input_ids": ov.PartialShape([BATCH_SIZE, seq_dim]),  # type: ignore[attr-defined]
            "attention_mask": ov.PartialShape(  # type: ignore[attr-defined]
                [BATCH_SIZE, seq_dim]
            ),
            "token_type_ids": ov.PartialShape(  # type: ignore[attr-defined]
                [BATCH_SIZE, seq_dim]
            ),
        })
        self.model[0].auto_model.compile()

        self.sources: List[MinimalSource] = []
        self._sources_by_id: Dict[int, MinimalSource] = {}
        try:
            self.load()
        except FileNotFoundError:
            dimension: int = self.model.get_embedding_dimension()
            base_index = faiss.IndexFlatIP(dimension)
            self.index = faiss.IndexIDMap(base_index)

    def create_index(
        self,
        chunks: List[MinimalSource],
        deleted_paths: Set[str],
        batch_size: int = BATCH_SIZE,
    ) -> None:
        """Encodes chunks into dense vectors and adds them to FAISS index"""
        if not chunks and not deleted_paths:
            return

        updated_files = set(c.file_path for c in chunks)
        ids_to_remove = [
            s.faiss_custom_id
            for s in self.sources
            if s.file_path in updated_files or s.file_path in deleted_paths
        ]

        if ids_to_remove:
            self.index.remove_ids(np.array(ids_to_remove, dtype=np.int64))
            self.sources = [
                s
                for s in self.sources
                if s.file_path not in updated_files
                and s.file_path not in deleted_paths
            ]

        if chunks:
            self.sources.extend(chunks)
            texts: List[str] = [c.content for c in chunks]
            new_ids: np.ndarray = np.array(
                [c.faiss_custom_id for c in chunks], dtype=np.int64
            )

            embeddings: np.ndarray = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).astype(np.float32)

            self.index.add_with_ids(embeddings, new_ids)

        self._reindex_sources()

    def _reindex_sources(self) -> None:
        """Rebuilds the faiss_custom_id -> MinimalSource lookup that
        search() and search_batch() use to map FAISS's returned ids back
        to sources, so it stays in sync with self.sources."""
        self._sources_by_id = {
            s.faiss_custom_id: s
            for s in self.sources
            if s.faiss_custom_id is not None
        }

    def search(self, query: str, k: int = 10) -> List[MinimalSource]:
        """Embeds a single query and returns its top-k nearest chunks,
        ranked closest first.

        Parameters:
            query (str): The natural-language question to search for.
            k (int): Maximum number of sources to return.
        """
        return self.search_batch([query], k)[0]

    def search_batch(
        self, queries: List[str], k: int = 10
    ) -> List[List[MinimalSource]]:
        """Embeds many queries in one batched encode + one FAISS search
        call, and returns each query's top-k nearest chunks in the same
        order as `queries`.

        Parameters:
            queries (List[str]): The natural-language questions to search for.
            k (int): Maximum number of sources to return per query.
        """
        if not queries or k <= 0 or self.index.ntotal == 0:
            return [[] for _ in queries]

        embeddings: np.ndarray = self.model.encode(
            queries,
            batch_size=BATCH_SIZE,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        top_k = min(k, self.index.ntotal)
        _, id_rows = self.index.search(embeddings, top_k)

        return [
            [
                self._sources_by_id[source_id]
                for source_id in id_row
                if source_id != -1 and source_id in self._sources_by_id
            ]
            for id_row in id_rows
        ]

    def save(
        self, index_path: str = FAISS_INDEX, sources_path: str = FAISS_SOURCES
    ) -> None:
        os.makedirs(os.path.dirname(index_path), exist_ok=True)

        faiss.write_index(self.index, index_path)

        serialized_sources = [s.model_dump() for s in self.sources]
        with open(sources_path, "w", encoding="utf-8") as f:
            json.dump(serialized_sources, f, indent=4)

    def load(
        self,
        index_path: str = FAISS_INDEX,
        sources_path: str = FAISS_SOURCES,
    ) -> None:
        """Loads the FAISS index and the source metadata from disk.

        Raises:
            FileNotFoundError: If no index was saved yet.
            CorruptedIndex: If the saved files cannot be read back.
        """
        if not os.path.exists(index_path) or not os.path.exists(sources_path):
            raise FileNotFoundError("Vector index files missing.")

        try:
            self.index = faiss.read_index(index_path)
        except RuntimeError:
            raise CorruptedIndex(
                f"{index_path} is corrupted, run `index` again to rebuild it."
            )
        try:
            with open(sources_path, "r", encoding="utf-8") as f:
                raw_sources = json.load(f)
            self.sources = [MinimalSource(**item) for item in raw_sources]
        except (ValueError, TypeError, ValidationError):
            raise CorruptedIndex(
                f"{sources_path} is corrupted, run `index` again to "
                "rebuild it."
            )
        self._reindex_sources()
