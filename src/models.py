from pydantic import BaseModel, Field
from typing import Dict, List, Optional
import uuid
import hashlib


def make_source_id(
    file_path: str, first_character_index: int, last_character_index: int
) -> int:
    """Derives a stable 63-bit ID for a chunk from its location.

    The same span of the same file always gets the same ID, so the ID
    can join the lexical and semantic indexes without touching the
    filesystem. 63 bits keeps it a valid positive FAISS int64.

    Args:
        file_path: Corpus path of the chunk's file.
        first_character_index: Start of the chunk's span.
        last_character_index: End of the chunk's span.

    Returns:
        A non-negative integer below 2**63.
    """
    key = f"{file_path}:{first_character_index}:{last_character_index}"
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big") & 0x7FFFFFFFFFFFFFFF


class MinimalSource(BaseModel):
    """A span of a corpus file. `content` and `faiss_custom_id` are
    filled in by the indexer and stay empty on ground-truth sources."""

    file_path: str
    first_character_index: int
    last_character_index: int
    content: str = ""
    faiss_custom_id: Optional[int] = None


class UnansweredQuestion(BaseModel):
    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    rag_questions: List[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    answer: str


class StudentSearchResults(BaseModel):
    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    search_results: List[MinimalAnswer]
    k: int


class IndexCache(BaseModel):
    """Indexing state used to re-index only the files that changed."""

    file_hashes: Dict[str, str] = {}
    max_chunk_size: Optional[int] = None
    index_version: Optional[int] = None
