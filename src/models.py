"""Pydantic models exchanged between the pipeline's stages."""
import hashlib
import re
import uuid
from typing import Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


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
    """A question, without its reference sources."""

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """A question with its reference sources and answer."""

    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """A dataset of questions."""

    rag_questions: List[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """The sources retrieved for one question."""

    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """The sources retrieved for one question, and its answer."""

    answer: str


class StudentSearchResults(BaseModel):
    """Output of search_dataset."""

    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """Output of answer and answer_dataset."""

    search_results: List[MinimalAnswer]
    k: int


class IndexCache(BaseModel):
    """Indexing state used to re-index only the files that changed."""

    file_hashes: Dict[str, str] = {}
    max_chunk_size: Optional[int] = None
    index_version: Optional[int] = None


# Source URL grammar for RagConfig (remote sources: not wired yet).
# No leading "-", no ":" (so no user:pass@).
USER = r"[A-Za-z0-9_][A-Za-z0-9._-]{0,63}"
LABEL = r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?"
# Hostname, IPv4, or [IPv6].
HOST = rf"(?:\[[0-9A-Fa-f:.]+\]|{LABEL}(?:\.{LABEL})*)"
PORT = r"(?::\d{1,5})?"
# One path segment: no spaces, backslashes, control characters.
SEG = r"[^\s/\\\x00-\x1f]+"
PATH = rf"(?:/{SEG})*/?"
# No ".." segment anywhere.
NO_DOTDOT = r"(?!.*(?:^|[/:])\.\.(?:/|$))"

URL_RE = re.compile(rf"""^{NO_DOTDOT}(?:
    (?:file://)?(?:/|\./|~/)(?:{SEG})?{PATH}                # local
  | sftp://(?:{USER}@)?{HOST}{PORT}{PATH}                   # sftp
  | smb://(?:(?:{USER};)?{USER}@)?{HOST}{PORT}/{SEG}{PATH}  # smb
  | ssh://(?:{USER}@)?{HOST}{PORT}/{SEG}{PATH}              # git+ssh
  | https://(?:{USER}@)?{HOST}{PORT}/{SEG}{PATH}            # git+https
  | rsync://(?:{USER}@)?{HOST}{PORT}/{SEG}{PATH}            # rsync
  | {USER}@{HOST}:(?!//)(?:{SEG})?{PATH}                    # scp-like
)$""", re.VERBOSE)


class RagConfig(BaseModel):
    """Configuration for the RAG system."""

    source_path: str = "./data/raw"
    url_type: Optional[str] = None
    subdir: Optional[str] = None
    # "copy" would shadow BaseModel.copy().
    copy_source: bool = True
    key_file_path: Optional[str] = None
    # SSH KEY (SFTP/GIT-SSH/RSYNC)
    known_hosts_file_path: Optional[str] = "~/.ssh/known_hosts"
    username: Optional[str] = None
    password_env: Optional[str] = None
    # GIT
    token_env: Optional[str] = None
    branch: Optional[str] = None
    # SMB
    domain: Optional[str] = None

    model_config = ConfigDict(validate_assignment=True)
