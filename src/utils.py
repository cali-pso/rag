import os
import hashlib
from typing import Dict, List
from src.models import (
    IndexCache,
)

CACHE_PATH = "data/processed/index_cache.json"


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
        with open(CACHE_PATH, "r") as f:
            return IndexCache.model_validate_json(f.read())
    except Exception:
        return IndexCache()


def save_cache(cache: IndexCache) -> None:
    """Saves the current indexing state."""
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        f.write(cache.model_dump_json(indent=4))


def sort_files(
    root_path: str, cache: IndexCache
) -> (Dict[str, List[str]], IndexCache):
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
    with open(dataset_path, "r") as f:
        return RagDataset.model_validate_json(f.read())
