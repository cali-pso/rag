import os
import time
import fire
import hashlib
from typing import Dict, List
from src.lexical_indexer import load_index
from src.bm25retriever import BM25Retriever
from src.models import (
    IndexCache,
    UnansweredQuestion,
    MinimalSearchResults,
    StudentSearchResults,
    RagDataset,
)

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

LEXICAL_INDEX_PATH = "data/processed/lexical_index.json"
SINGLE_QUERY_RESULT_PATH = (
    "data/output/search_results/single_query/search_results.json"
)
BATCH_QUERY_RESULT_PATH = (
    "data/output/search_results/single_query/search_dataset_results.json"
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


def index(max_chunk_size: int = 2000) -> None:
    from src.lexical_indexer import LexicalIndexer
    from src.vector_indexer import VectorIndexer

    print("Loading cache...")
    start = time.time()
    cache = load_cache()
    print("Looking up files that need indexing...")
    paths, deleted_paths = sort_files("data/raw", cache)
    all_paths = paths["md_paths"] + paths["py_paths"]

    print(f"Indexing {len(all_paths)} files...")
    lexical_indexer = LexicalIndexer()
    lexical_indexer.create_index(paths, deleted_paths)
    sources_dict = lexical_indexer.chunks
    sources = []
    for file in sources_dict.keys():
        if file in all_paths:
            for source in sources_dict[file]:
                sources.append(source)
    print(
        f"Indexed as {len(sources)} chunks",
        f"in {(time.time() - start):.3f}s.",
    )
    print("Building vector index from chunks...")
    vector_indexer = VectorIndexer()
    vector_indexer.create_index(sources, deleted_paths)
    vector_indexer.save()
    print("Saving results...")
    save_cache(cache)
    print("Done.")
    print(f"Built lexical + semantic index in {(time.time() - start):.3f}s.")


def search(query: str, k: int = 10) -> None:
    print(f"Processing query: {query}")
    question = UnansweredQuestion(question=query)
    start = time.time()
    retriever = BM25Retriever()
    cache = load_cache()
    _, deleted_paths = sort_files("data/raw", cache)
    sources_dict = load_index(deleted_paths)
    sources = []
    for file in sources_dict:
        for source in sources_dict[file]:
            sources.append(source)
    retriever.fit(sources)
    answer = retriever.search(question.question, k)
    print(f"Retrieved {len(answer)} sources in {(time.time() - start):.3f}s")
    search_result = MinimalSearchResults(
        question_id=question.question_id,
        question=question.question,
        retrieved_sources=answer,
    )
    student_search_result = StudentSearchResults(
        search_results=[search_result], k=k
    )
    print("Writing results to disk...")
    os.makedirs(os.path.dirname(SINGLE_QUERY_RESULT_PATH), exist_ok=True)
    with open(SINGLE_QUERY_RESULT_PATH, "w") as f:
        f.write(student_search_result.model_dump_json())
    print("Done.")


def _load_dataset(dataset_path: str) -> RagDataset:
    with open(dataset_path, "r") as f:
        return RagDataset.model_validate_json(f.read())


def search_dataset(
    dataset_path: str,
    save_directory: str,
    k: int = 10,
):
    dataset = _load_dataset(dataset_path)
    retriever = BM25Retriever()
    cache = load_cache()
    _, deleted_paths = sort_files("data/raw", cache)
    sources_dict = load_index(deleted_paths)
    sources = []
    for file in sources_dict:
        for source in sources_dict[file]:
            sources.append(source)
    retriever.fit(sources)
    search_results = []
    print(f"Treating {len(dataset.rag_questions)}")
    for question in dataset.rag_questions:
        answer = retriever.search(question.question, k)
        search_result = MinimalSearchResults(
            question_id=question.question_id,
            question=question.question,
            retrieved_sources=answer,
        )
        search_results.append(search_result)
    student_search_results = StudentSearchResults(
        search_results=search_results, k=k
    )
    print("Writing results to disk...")
    os.makedirs(
        os.path.dirname(save_directory + "/student_search_results.json"),
        exist_ok=True,
    )
    with open(save_directory + "/student_search_results.json", "w") as f:
        f.write(student_search_results.model_dump_json())
    print("Done.")


if __name__ == "__main__":
    fire.Fire()
