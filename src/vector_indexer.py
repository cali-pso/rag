import os
import json
from typing import List
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from src.models import MinimalSource

MODEL_NAME = "./openvino/openvino_weight_only_int8_quantized"
BATCH_SIZE = 16
MAX_SEQ_LENGTH = 128
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
        # Force static shape and lock model into CPU memory
        ov_model = self.model[0].auto_model.model
        ov_model.reshape(
            {
                "input_ids": [BATCH_SIZE, MAX_SEQ_LENGTH],
                "attention_mask": [BATCH_SIZE, MAX_SEQ_LENGTH],
                "token_type_ids": [BATCH_SIZE, MAX_SEQ_LENGTH],
            }
        )
        self.model[0].auto_model.compile()

        self.sources: List[MinimalSource] = []
        try:
            self.load()
        except FileNotFoundError:
            pass

    def create_index(
        self, chunks: List[MinimalSource], batch_size: int = BATCH_SIZE
    ) -> None:
        """Encodes chunks into dense vectors and adds them to FAISS index"""
        if not chunks:
            return

        self.sources = chunks
        texts: List[str] = [c.content for c in chunks]

        embeddings: np.ndarray = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

        dimension: int = self.model.get_embedding_dimension()
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)

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
        index_path: str = "data/processed/faiss_index.bin",
        sources_path: str = "data/processed/faiss_sources.json",
    ) -> None:
        """
        Loads the FAISS index and the source metadata from disk.
        """
        if not os.path.exists(index_path) or not os.path.exists(sources_path):
            raise FileNotFoundError("Vector index files missing.")

        self.index = faiss.read_index(index_path)
        with open(sources_path, "r", encoding="utf-8") as f:
            raw_sources = json.load(f)
            self.sources = [MinimalSource(**item) for item in raw_sources]
