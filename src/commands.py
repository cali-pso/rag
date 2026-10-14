import functools
import os
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple
from tqdm import tqdm
from src.lexical_indexer import load_index, LEXICAL_INDEX_PATH
from src.bm25retriever import BM25Retriever
from src.hybrid_retriever import HybridRetriever
from src.evaluator import Evaluator
from src.errors import IndexNotFound, InvalidArgument
from src.utils import (
    NO_SOURCES_ANSWER,
    as_path,
    as_query,
    check_k,
    load_cache,
    load_dataset,
    read_json_model,
    save_cache,
    sort_files,
)
from src.models import (
    UnansweredQuestion,
    MinimalSearchResults,
    StudentSearchResults,
    MinimalAnswer,
    StudentSearchResultsAndAnswer,
)

if TYPE_CHECKING:
    from src.oracle import Oracle

# The embedding model (torch, sentence-transformers, OpenVINO) and Qwen
# are only imported by the functions that need them: a command rejected
# on its arguments answers in about a second instead of after ~10 s of
# imports.

RAW_DATA_PATH = "data/raw"
SINGLE_QUERY_SEARCH_DIR = "data/output/search_results/single_query"
SINGLE_QUERY_ANSWER_DIR = "data/output/search_results_and_answer/single_query"
SINGLE_QUERY_FILE_NAME = "single_query.json"
MAX_CHUNK_SIZE_LIMIT = 2000
MIN_CHUNK_SIZE = 100
# Bump whenever the on-disk index format changes: a mismatch with the
# cache forces a full rebuild instead of mixing old and new chunks.
# 2: chunk IDs derived from the chunk's span instead of the file mtime.
# 3: only Markdown chunks are embedded in the semantic index.
# 4: the file path is part of the text searched by BM25 and embedded.
INDEX_VERSION = 4
# Questions sent to the model per generate() call: every prompt in a
# batch is padded to the longest one, so large batches of ~5k-token
# prompts blow up memory on a CPU-only machine.
ANSWER_BATCH_SIZE = 1
# Questions searched per step of search_dataset's progress bar; the
# semantic side still embeds each step's questions in one batch.
SEARCH_BATCH_SIZE = 32
# Retriever kept in memory with the modification times of the index
# files it was built from, so a long-running process (the HTTP API)
# reuses it and rebuilds it only after the index changed on disk.
_RETRIEVER_CACHE: List[Tuple[Tuple[int, ...], HybridRetriever]] = []


def _write_json(save_directory: str, file_name: str, json_str: str) -> str:
    """Writes json_str to save_directory/file_name, creating the directory
    if needed, and returns the written path."""
    os.makedirs(save_directory, exist_ok=True)
    out_path = os.path.join(save_directory, file_name)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json_str)
    return out_path


def _index_files() -> Tuple[str, ...]:
    """Paths of every file making up the persisted index."""
    from src.vector_indexer import FAISS_INDEX, FAISS_SOURCES

    return (LEXICAL_INDEX_PATH, FAISS_INDEX, FAISS_SOURCES)


def _build_retriever() -> HybridRetriever:
    """Loads the persisted lexical and semantic indexes and wraps them in a
    HybridRetriever, reusing the one already in memory if the index files
    have not changed since it was built.

    Raises:
        IndexNotFound: If `index` has not been run yet.
    """
    from src.vector_indexer import VectorIndexer

    paths = _index_files()
    if not all(os.path.isfile(path) for path in paths):
        raise IndexNotFound(
            "no index found under data/processed/, run `index` first."
        )
    stamp = tuple(os.stat(path).st_mtime_ns for path in paths)
    if _RETRIEVER_CACHE and _RETRIEVER_CACHE[0][0] == stamp:
        return _RETRIEVER_CACHE[0][1]
    # The saved index is the snapshot of the last `index` run: files
    # deleted since then are dropped by re-running `index`, not here.
    sources_dict = load_index(set())
    sources = [s for file in sources_dict for s in sources_dict[file]]
    bm25 = BM25Retriever()
    bm25.fit(sources)
    retriever = HybridRetriever(bm25=bm25, vector_indexer=VectorIndexer())
    _RETRIEVER_CACHE[:] = [(stamp, retriever)]
    return retriever


@functools.lru_cache(maxsize=1)
def _get_oracle() -> "Oracle":
    """Loads Qwen3-0.6B once per process (exporting it on first use)."""
    from src.oracle import Oracle

    return Oracle()


def _clear_index() -> None:
    """Removes every persisted index file so the next build starts from
    scratch."""
    for path in _index_files():
        if os.path.exists(path):
            os.remove(path)


def index(
    max_chunk_size: int = MAX_CHUNK_SIZE_LIMIT,
) -> Dict[str, str | int | float]:
    """Ingests data/raw/ and builds the lexical and semantic indexes under
    data/processed/.

    Args:
        max_chunk_size: Maximum width, in characters, of a chunk's span.
    """
    from src.lexical_indexer import LexicalIndexer
    from src.vector_indexer import VectorIndexer

    if (
        isinstance(max_chunk_size, bool)
        or not isinstance(max_chunk_size, int)
        or not MIN_CHUNK_SIZE <= max_chunk_size <= MAX_CHUNK_SIZE_LIMIT
    ):
        raise InvalidArgument(
            "--max_chunk_size must be an integer between "
            f"{MIN_CHUNK_SIZE} and {MAX_CHUNK_SIZE_LIMIT}, "
            f"got {max_chunk_size!r}."
        )
    if not os.path.isdir(RAW_DATA_PATH):
        raise IndexNotFound(
            f"corpus directory {RAW_DATA_PATH}/ not found: put the vLLM "
            "repository there before indexing."
        )

    print("Loading cache...")
    start = time.time()
    cache = load_cache()
    if (
        cache.max_chunk_size != max_chunk_size
        or cache.index_version != INDEX_VERSION
    ):
        # Every existing chunk was cut at a different size, or saved in
        # an older format: the per-file hashes can't tell us that, so
        # throw away the whole index.
        if cache.max_chunk_size not in (None, max_chunk_size):
            print(
                f"Chunk size changed ({cache.max_chunk_size} -> "
                f"{max_chunk_size}), rebuilding the whole index..."
            )
        elif cache.max_chunk_size is not None:
            print("Index format changed, rebuilding the whole index...")
        _clear_index()
        cache.file_hashes = {}
        cache.max_chunk_size = max_chunk_size
        cache.index_version = INDEX_VERSION
    print("Looking up files that need indexing...")
    paths, deleted_paths = sort_files(RAW_DATA_PATH, cache)
    all_paths = paths["md_paths"] + paths["py_paths"]

    print(f"Indexing {len(all_paths)} files...")
    lexical_indexer = LexicalIndexer()
    lexical_indexer.create_index(paths, deleted_paths, max_chunk_size)
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
    print("Building vector index from the Markdown chunks...")
    vector_indexer = VectorIndexer()
    vector_indexer.create_index(sources, deleted_paths)
    vector_indexer.save()
    print("Saving results...")
    save_cache(cache)
    print(f"Built lexical + semantic index in {(time.time() - start):.3f}s.")
    print("Ingestion complete! Indices saved under data/processed/")
    return {
        "status": "done",
        "time_elapsed": time.time() - start,
        "files_indexed": len(all_paths),
        "chunks": len(sources),
    }


def search(query: Any, k: Any = 10) -> Dict[str, Any]:
    """Retrieves the top-k sources for a single query and prints them.

    Args:
        query: The question to search for.
        k: Number of sources to retrieve.
    """
    query = as_query(query)
    k = check_k(k)
    if not query.strip():
        print("Warning: empty query, nothing to search for.")
    print(f"Processing query: {query}")
    bar = tqdm(range(3))
    question = UnansweredQuestion(question=query)
    start = time.time()
    bar.update(1)
    bar.refresh()
    retriever = _build_retriever()
    bar.update(1)
    bar.refresh()
    sources = retriever.search(question.question, k)
    bar.update(1)
    bar.refresh()
    print(f"Retrieved {len(sources)} sources in {(time.time() - start):.3f}s")
    for rank, source in enumerate(sources, start=1):
        print(
            f"{rank}. {source.file_path} "
            f"[{source.first_character_index}:"
            f"{source.last_character_index}]"
        )
    return MinimalSearchResults(
        question_id=question.question_id,
        question=question.question,
        retrieved_sources=sources,
    ).model_dump()


def search_dataset(
    dataset_path: Any,
    save_directory: Any = "data/output/search_results/",
    k: Any = 10,
) -> None:
    """Runs search over every question of a dataset and writes a
    StudentSearchResults JSON file named after the dataset file.

    Args:
        dataset_path: Path to a RagDataset JSON file.
        save_directory: Directory the results file is written to.
        k: Number of sources to retrieve per question.
    """
    dataset_path = as_path("dataset_path", dataset_path)
    save_directory = as_path("save_directory", save_directory)
    k = check_k(k)
    # Fail on an unusable output directory now, not after the search.
    os.makedirs(save_directory, exist_ok=True)
    dataset = load_dataset(dataset_path)
    retriever = _build_retriever()

    print(f"Treating {len(dataset.rag_questions)} questions")
    queries = [question.question for question in dataset.rag_questions]
    batch_results: List[List[Any]] = []
    with tqdm(total=len(queries), desc="Searching") as progress:
        for i in range(0, len(queries), SEARCH_BATCH_SIZE):
            end = i + SEARCH_BATCH_SIZE
            batch_results.extend(retriever.search_batch(queries[i:end], k))
            progress.update(len(queries[i:end]))

    search_results = [
        MinimalSearchResults(
            question_id=question.question_id,
            question=question.question,
            retrieved_sources=sources,
        )
        for question, sources in zip(dataset.rag_questions, batch_results)
    ]
    student_search_results = StudentSearchResults(
        search_results=search_results, k=k
    )
    out_path = _write_json(
        save_directory,
        os.path.basename(dataset_path),
        student_search_results.model_dump_json(),
    )
    print(f"Saved student_search_results to {out_path}")


def answer(
    query: Any, k: Any = 10, save_directory: Any = SINGLE_QUERY_ANSWER_DIR
) -> Optional[Dict[str, Any]]:
    """Retrieves the top-k sources for a single query, generates an answer
    from them, prints it and saves a StudentSearchResultsAndAnswer JSON
    file.

    Args:
        query: The question to answer.
        k: Number of sources to retrieve and pass to the model.
        save_directory: Directory the answer file is written to.
    """
    query = as_query(query)
    k = check_k(k)
    save_directory = as_path("save_directory", save_directory)
    if not query.strip():
        print("Warning: empty query, there is nothing to answer.")
        return None
    # Fail on an unusable output directory now, not after generation.
    os.makedirs(save_directory, exist_ok=True)

    print(f"Processing query: {query}")
    question = UnansweredQuestion(question=query)
    retriever = _build_retriever()
    result = MinimalSearchResults(
        question_id=question.question_id,
        question=question.question,
        retrieved_sources=retriever.search(question.question, k),
    )
    minimal_answer = MinimalAnswer(
        question_id=result.question_id,
        question=result.question,
        retrieved_sources=result.retrieved_sources,
        # Nothing retrieved (e.g. k=0): no context to ground an answer in,
        # so don't even load the model.
        answer=(
            _get_oracle().answer(result)
            if result.retrieved_sources
            else NO_SOURCES_ANSWER
        ),
    )
    print(minimal_answer.answer)
    out_path = _write_json(
        save_directory,
        SINGLE_QUERY_FILE_NAME,
        StudentSearchResultsAndAnswer(
            search_results=[minimal_answer], k=k
        ).model_dump_json(),
    )
    print(f"Saved answer to {out_path}")
    return minimal_answer.model_dump()


def answer_dataset(
    student_search_results_path: Any, save_directory: Any
) -> None:
    """Generates an answer for every question of a StudentSearchResults
    file and writes a StudentSearchResultsAndAnswer JSON file with the
    same name.

    Args:
        student_search_results_path: Output of search_dataset.
        save_directory: Directory the answers file is written to.
    """
    student_search_results_path = as_path(
        "student_search_results_path", student_search_results_path
    )
    save_directory = as_path("save_directory", save_directory)
    student_search_results = read_json_model(
        student_search_results_path, StudentSearchResults
    )
    # Fail on an unusable output directory now, not after generation.
    os.makedirs(save_directory, exist_ok=True)

    results = student_search_results.search_results
    print(f"Loaded {len(results)} questions")
    model = _get_oracle()
    answers = []
    with tqdm(total=len(results), desc="Answering") as progress:
        for i in range(0, len(results), ANSWER_BATCH_SIZE):
            end = i + ANSWER_BATCH_SIZE
            batch = results[i:end]
            for result, text in zip(batch, model.answer_batch(batch)):
                answers.append(
                    MinimalAnswer(
                        question_id=result.question_id,
                        question=result.question,
                        retrieved_sources=result.retrieved_sources,
                        answer=text,
                    )
                )
            progress.update(len(batch))
    print(f"Processed {len(answers)} of {len(results)} questions")
    out_path = _write_json(
        save_directory,
        os.path.basename(student_search_results_path),
        StudentSearchResultsAndAnswer(
            search_results=answers, k=student_search_results.k
        ).model_dump_json(),
    )
    print(f"Saved student_search_results_and_answer to {out_path}")


def evaluate(student_search_results_path: Any, dataset_path: Any) -> None:
    """Prints recall@k of a StudentSearchResults file against a ground-truth
    AnsweredQuestions dataset.

    Args:
        student_search_results_path: Output of search_dataset.
        dataset_path: Ground-truth AnsweredQuestions dataset.
    """
    student_search_results_path = as_path(
        "student_search_results_path", student_search_results_path
    )
    dataset_path = as_path("dataset_path", dataset_path)
    evaluator = Evaluator()
    recalls = evaluator.calculate_recall_at_k(
        student_search_results_path, dataset_path
    )
    if evaluator.without_sources:
        print(
            f"Warning: skipped {evaluator.without_sources} question(s) "
            "without ground-truth sources (is --dataset_path an "
            "AnsweredQuestions file?)."
        )
    if evaluator.unmatched:
        print(
            f"Warning: skipped {evaluator.unmatched} question(s) with no "
            "matching question_id in the student results."
        )
    print("Evaluation Results")
    print("=" * 40)
    print("  ".join(f"{name}: {value:.3f}" for name, value in recalls.items()))
