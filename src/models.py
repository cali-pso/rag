from pydantic import BaseModel
from typing import Dict


class MinimalSource(BaseModel):
    file_path: str
    first_character_index: int
    last_character_index: int
    content: str


class IndexCache(BaseModel):
    file_hashes: Dict[str, str] = {}
