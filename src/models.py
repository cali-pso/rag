from pydantic import BaseModel
from typing import Dict, List


class MinimalSource(BaseModel):
    file_path: str
    first_character_index: int
    last_character_index: int
    content: str


class IndexCache(BaseModel):
    file_hashes: Dict[str, str] = {}


class MinimalSearchResults(BaseModel):
    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    answer: str
