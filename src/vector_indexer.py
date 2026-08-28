import os
import json
from typing import List
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from src.models import MinimalSource

MODEL_NAME = "all-MiniLM-L6-v2"
OPENVINO_FILE_NAME = "openvino/openvino_model_qint8_quantized.xml"
FAISS_INDEX = "data/processed/faiss_index.bin"
FAISS_SOURCES = "data/processed/faiss_sources.json"


class VectorIndexer:
    def __init__(self, model_name: str = MODEL_NAME) -> None:
        """
        Initializes the sentence transformer model and the underlying FAISS index.

        Parameters:
            model_name (str): HuggingFace model identifier for sentence embeddings.
        """
        self.model: SentenceTransformer = SentenceTransformer(
            model_name,
            backend="openvino",
            model_kwargs={"file_name": OPENVINO_FILE_NAME},
        )
        self.dimension: int = self.model.get_embedding_dimension()
        self.index: faiss.IndexFlatIP = faiss.IndexFlatIP(self.dimension)
        self.sources: List[MinimalSource] = []
        # try:
        #    self.load()
        # except FileNotFoundError:
        #    pass

    def create_index(
        self, chunks: List[MinimalSource], batch_size: int = 16
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

        self.index = faiss.IndexFlatIP(self.dimension)
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
