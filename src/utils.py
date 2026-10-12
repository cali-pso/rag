import os
import hashlib
from typing import Any, Dict, List, Set, Tuple, Type, TypeVar
from pydantic import BaseModel, ValidationError
from src.errors import InvalidArgument, InvalidInputFile
from src.models import (
    IndexCache,
    RagDataset,
)

CACHE_PATH = "data/processed/index_cache.json"

ModelT = TypeVar("ModelT", bound=BaseModel)


def check_k(k: Any) -> int:
    """Validates a --k value coming from Fire.

    Fire guesses argument types, so `--k abc` arrives as a str, `--k 2.5`
    as a float and a bare `--k` as True (bool is a subclass of int).

    Args:
        k: The raw value received from the command line.

    Returns:
        k, once known to be a non-negative int.

    Raises:
        InvalidArgument: If k is not a non-negative integer.
    """
    if isinstance(k, bool) or not isinstance(k, int) or k < 0:
        raise InvalidArgument(
            f"--k must be a non-negative integer, got {k!r}."
        )
    return k


def as_path(name: str, value: Any) -> str:
    """Validates a path argument coming from Fire.

    `--dataset_path 3` arrives as the int 3, and open(3) would silently
    open file descriptor 3 instead of a file named "3"; a bare
    `--save_directory` arrives as True.

    Args:
        name: The argument name, used in the error message.
        value: The raw value received from the command line.

    Returns:
        The value as a non-empty path string.

    Raises:
        InvalidArgument: If no usable path was given.
    """
    if isinstance(value, bool) or value is None or str(value) == "":
        raise InvalidArgument(f"--{name} expects a path, got {value!r}.")
    return str(value)


def as_query(value: Any) -> str:
    """Turns a query coming from Fire back into text (`search 2024`
    arrives as the int 2024)."""
    return str(value)


def read_json_model(path: str, model: Type[ModelT]) -> ModelT:
    """Reads a user-supplied JSON file and validates it against a model.

    Args:
        path: The JSON file to read.
        model: The pydantic model the file must follow.

    Returns:
        The validated model instance.

    Raises:
        OSError: If the file cannot be opened (missing, directory, ...).
        InvalidInputFile: If the file is not valid JSON for the model.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return model.model_validate_json(f.read())
    except UnicodeDecodeError:
        raise InvalidInputFile(f"{path} is not a UTF-8 text file.")
    except ValidationError as e:
        first = e.errors()[0]
        location = ".".join(str(part) for part in first["loc"]) or "root"
        raise InvalidInputFile(
            f"{path} is not a valid {model.__name__} file "
            f"({e.error_count()} error(s), first at {location}: "
            f"{first['msg']})."
        )


def get_file_hash(filepath: str) -> str:
    """Computes the SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


def load_cache() -> IndexCache:
    """Loads the previous indexing state if it exists."""
    try:
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            return IndexCache.model_validate_json(f.read())
    except Exception:
        # Missing or unreadable cache: starting from an empty one only
        # costs a full re-index, which is the correct recovery anyway.
        return IndexCache()


def save_cache(cache: IndexCache) -> None:
    """Saves the current indexing state."""
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        f.write(cache.model_dump_json(indent=4))


def sort_files(
    root_path: str, cache: IndexCache
) -> Tuple[Dict[str, List[str]], Set[str]]:
    """Finds the files under root_path that changed since the last run.

    Updates the hashes in cache as a side effect.

    Args:
        root_path: The corpus directory.
        cache: The previous indexing state.

    Returns:
        The new or modified Markdown and Python paths, and the set of
        previously indexed paths that no longer exist.
    """
    md_paths = []
    py_paths = []
    present_paths = set()

    for root, _, files in os.walk(root_path):
        for file in files:
            full_path = os.path.join(root, file)
            if file.endswith(".md"):
                current_hash = get_file_hash(full_path)
                if cache.file_hashes.get(full_path) != current_hash:
                    cache.file_hashes[full_path] = current_hash
                    md_paths.append(full_path)
            elif file.endswith(".py"):
                current_hash = get_file_hash(full_path)
                if cache.file_hashes.get(full_path) != current_hash:
                    cache.file_hashes[full_path] = current_hash
                    py_paths.append(full_path)
            present_paths.add(full_path)
    cached_paths = set(cache.file_hashes.keys())
    deleted_paths = cached_paths.difference(present_paths)
    for file in deleted_paths:
        cache.file_hashes.pop(file)
    return {"md_paths": md_paths, "py_paths": py_paths}, deleted_paths


def load_dataset(dataset_path: str) -> RagDataset:
    """Reads and validates a RagDataset JSON file."""
    return read_json_model(dataset_path, RagDataset)
