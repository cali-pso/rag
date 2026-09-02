from pydantic import BaseModel, model_validator
from typing import Dict, List, Optional
import os
import hashlib
from random import random


class MinimalSource(BaseModel):
    file_path: str
    first_character_index: int
    last_character_index: int
    content: str
    faiss_custom_id: Optional[int] = None

    @model_validator(mode="after")
    def generate_faiss_custom_id(self):
        """Generates a stable, unique 64-bit signed integer ID after init."""
        mtime = os.path.getmtime(self.file_path)
        index = self.first_character_index
        unique_string = (
            f"{self.file_path}_{index if index != -1 else random()}_{mtime}"
        )
        hash_bytes = hashlib.sha256(unique_string.encode("utf-8")).digest()
        self.faiss_custom_id = (
            int.from_bytes(hash_bytes[:8], byteorder="big")
            & 0x7FFFFFFFFFFFFFFF
        )

        return self


class IndexCache(BaseModel):
    file_hashes: Dict[str, str] = {}


class MinimalSearchResults(BaseModel):
    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    answer: str
