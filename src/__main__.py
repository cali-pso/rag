import os
import time
import fire
import hashlib
import json
from typing import Dict, List
from src.chunker import Chunker
from src.models import IndexCache

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
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "r") as f:
            return IndexCache(**json.load(f))
    return IndexCache()


def save_cache(cache: IndexCache) -> None:
    """Saves the current indexing state."""
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        f.write(cache.model_dump_json(indent=4))


def sort_files(root_path: str) -> Dict[str, List[str]]:
    md_paths = []
    py_paths = []
    cache = load_cache()

    for root, _, files in os.walk(root_path):
        for file in files:
            if file.endswith(".md"):
                full_path = os.path.join(root, file)
                current_hash = get_file_hash(full_path)
                if cache.file_hashes.get(full_path) != current_hash:
                    cache.file_hashes[full_path] = current_hash
                    md_paths.append(full_path)
            elif file.endswith(".py"):
                full_path = os.path.join(root, file)
                current_hash = get_file_hash(full_path)
                if cache.file_hashes.get(full_path) != current_hash:
                    cache.file_hashes[full_path] = current_hash
                    py_paths.append(full_path)
    save_cache(cache)
    return {"md_paths": md_paths, "py_paths": py_paths}


def index(max_chunk_size: int = 2000) -> None:
    start = time.time()
    paths = sort_files("data")
    chunker = Chunker()
    sources = []
    chunks = 0
    files = 0
    for file in paths["md_paths"]:
        for chunk in chunker.chunk_md(
            file_path=file, max_chunk_size=max_chunk_size
        ):
            chunks += 1
            sources.append(chunk)
        files += 1
    for file in paths["py_paths"]:
        for chunk in chunker.chunk_py(
            file_path=file, max_chunk_size=max_chunk_size
        ):
            chunks += 1
            sources.append(chunk)
        files += 1
    chunker.dump_index(sources)
    print(
        f"Indexed {files} files as {chunks} sources in {time.time() - start}s"
    )


if __name__ == "__main__":
    fire.Fire()
