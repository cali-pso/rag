import os
import time
import fire
import hashlib
from typing import Dict, List
from src.chunker import LexicalIndexer, load_index
from src.vector_indexer import VectorIndexer
from src.bm25retriever import BM25Retriever
from src.models import IndexCache

LEXICAL_INDEX_PATH = "data/processed/lexical_index.json"
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
    return {"md_paths": md_paths, "py_paths": py_paths}, cache


def index(max_chunk_size: int = 2000) -> None:
    start = time.time()
    cache = load_cache()
    paths, updated_cache = sort_files("data", cache)
    all_paths = paths["md_paths"] + paths["py_paths"]
    file_names = [s.split('/')[-1] for s in paths]

    lexical_indexer = LexicalIndexer()
    lexical_indexer.create_index(paths)
    sources_dict = lexical_indexer.chunks
    sources = []
    for file in sources_dict.keys():
        if file in all_paths:
            for source in sources_dict[file]:
                sources.append(source)

    vector_indexer = VectorIndexer()
    vector_indexer.create_index(sources)
    vector_indexer.save()

    save_cache(cache)
    print(f"Indexed {len(lexical_indexer.chunks)} in {time.time() - start}s")


def search(query: str, k: int = 10) -> None:
    start = time.time()
    retriever = BM25Retriever()
    sources_dict = load_index()
    sources = []
    for file in sources_dict:
        for source in sources_dict[file]:
            sources.append(source)
    retriever.fit(sources)
    answer = retriever.search(query, k)
    for source in answer:
        print(source)
        print()
        print()
    print(f"Retrieved {k} sources in {time.time() - start}s")


if __name__ == "__main__":
    fire.Fire()
