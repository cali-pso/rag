import os
import time
from tqdm import tqdm
from src.lexical_indexer import load_index, LEXICAL_INDEX_PATH
from src.bm25retriever import BM25Retriever
from src.vector_indexer import VectorIndexer, FAISS_INDEX, FAISS_SOURCES
from src.hybrid_retriever import HybridRetriever
from src.evaluator import Evaluator
from src.utils import load_cache, save_cache, sort_files, load_dataset
from src.models import (
    UnansweredQuestion,
    MinimalSearchResults,
    StudentSearchResults,
    MinimalAnswer,
    StudentSearchResultsAndAnswer,
)

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

RAW_DATA_PATH = "data/raw"
SINGLE_QUERY_SEARCH_DIR = "data/output/search_results/single_query"
SINGLE_QUERY_ANSWER_DIR = "data/output/search_results_and_answer/single_query"
SINGLE_QUERY_FILE_NAME = "single_query.json"
MAX_CHUNK_SIZE_LIMIT = 2000
MIN_CHUNK_SIZE = 100
# Questions sent to the model per generate() call: every prompt in a
# batch is padded to the longest one, so large batches of ~5k-token
# prompts blow up memory on a CPU-only machine.
ANSWER_BATCH_SIZE = 1


def _write_json(save_directory: str, file_name: str, json_str: str) -> str:
    """Writes json_str to save_directory/file_name, creating the directory
    if needed, and returns the written path."""
    os.makedirs(save_directory, exist_ok=True)
    out_path = os.path.join(save_directory, file_name)
    with open(out_path, "w") as f:
        f.write(json_str)
    return out_path


def _build_retriever() -> HybridRetriever:
    """Loads the persisted lexical and semantic indexes and wraps them in a
    HybridRetriever."""
    cache = load_cache()
    _, deleted_paths = sort_files(RAW_DATA_PATH, cache)
    sources_dict = load_index(deleted_paths) or {}
    sources = [s for file in sources_dict for s in sources_dict[file]]
    bm25 = BM25Retriever()
    bm25.fit(sources)
    return HybridRetriever(bm25=bm25, vector_indexer=VectorIndexer())


def _clear_index() -> None:
    """Removes every persisted index file so the next build starts from
    scratch."""
    for path in (LEXICAL_INDEX_PATH, FAISS_INDEX, FAISS_SOURCES):
        if os.path.exists(path):
            os.remove(path)


def index(max_chunk_size: int = MAX_CHUNK_SIZE_LIMIT) -> None:
    """Ingests data/raw/ and builds the lexical and semantic indexes under
    data/processed/.

    Args:
        max_chunk_size: Maximum width, in characters, of a chunk's span.
    """
    from src.lexical_indexer import LexicalIndexer

    if (
        isinstance(max_chunk_size, bool)
        or not isinstance(max_chunk_size, int)
        or not MIN_CHUNK_SIZE <= max_chunk_size <= MAX_CHUNK_SIZE_LIMIT
    ):
        print(
            "Error: --max_chunk_size must be an integer between "
            f"{MIN_CHUNK_SIZE} and {MAX_CHUNK_SIZE_LIMIT}, "
            f"got {max_chunk_size!r}."
        )
        return

    print("Loading cache...")
    start = time.time()
    cache = load_cache()
    if cache.max_chunk_size != max_chunk_size:
        # Every existing chunk was cut at a different size: the per-file
        # hashes can't tell us that, so throw away the whole index.
        if cache.max_chunk_size is not None:
            print(
                f"Chunk size changed ({cache.max_chunk_size} -> "
                f"{max_chunk_size}), rebuilding the whole index..."
            )
        _clear_index()
        cache.file_hashes = {}
        cache.max_chunk_size = max_chunk_size
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
    print("Building vector index from chunks...")
    vector_indexer = VectorIndexer()
    vector_indexer.create_index(sources, deleted_paths)
    vector_indexer.save()
    print("Saving results...")
    save_cache(cache)
    print(f"Built lexical + semantic index in {(time.time() - start):.3f}s.")
    print("Ingestion complete! Indices saved under data/processed/")


def search(query: str, k: int = 10) -> None:
    """Retrieves the top-k sources for a single query, prints them and
    saves them as a StudentSearchResults JSON file.

    Args:
        query: The question to search for.
        k: Number of sources to retrieve.
        save_directory: Directory the results file is written to.
    """
    print(f"Processing query: {query}")
    bar = tqdm(range(3))
    question = UnansweredQuestion(question=query)
    start = time.time()
    bar.update(1)
    bar.refresh()
    retriever = _build_retriever()
    bar.update(2)
    bar.refresh()
    sources = retriever.search(question.question, k)
    bar.update(3)
    bar.refresh()
    print(f"Retrieved {len(sources)} sources in {(time.time() - start):.3f}s")
    for rank, source in enumerate(sources, start=1):
        print(
            f"{rank}. {source.file_path} "
            f"[{source.first_character_index}:"
            f"{source.last_character_index}]"
        )


def search_dataset(
    dataset_path: str,
    save_directory: str,
    k: int = 10,
) -> None:
    """Runs search over every question of a dataset and writes a
    StudentSearchResults JSON file named after the dataset file.

    Args:
        dataset_path: Path to a RagDataset JSON file.
        save_directory: Directory the results file is written to.
        k: Number of sources to retrieve per question.
    """
    dataset = load_dataset(dataset_path)
    retriever = _build_retriever()

    print(f"Treating {len(dataset.rag_questions)} questions")
    queries = [question.question for question in dataset.rag_questions]
    batch_results = retriever.search_batch(queries, k)

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
    query: str, k: int = 10, save_directory: str = SINGLE_QUERY_ANSWER_DIR
) -> None:
    """Retrieves the top-k sources for a single query, generates an answer
    from them, prints it and saves a StudentSearchResultsAndAnswer JSON
    file.

    Args:
        query: The question to answer.
        k: Number of sources to retrieve and pass to the model.
        save_directory: Directory the answer file is written to.
    """
    from src.oracle import Oracle

    print(f"Processing query: {query}")
    question = UnansweredQuestion(question=query)
    retriever = _build_retriever()
    result = MinimalSearchResults(
        question_id=question.question_id,
        question=question.question,
        retrieved_sources=retriever.search(question.question, k),
    )
    model = Oracle()
    minimal_answer = MinimalAnswer(
        question_id=result.question_id,
        question=result.question,
        retrieved_sources=result.retrieved_sources,
        answer=model.answer(result),
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


def answer_dataset(
    student_search_results_path: str, save_directory: str
) -> None:
    """Generates an answer for every question of a StudentSearchResults
    file and writes a StudentSearchResultsAndAnswer JSON file with the
    same name.

    Args:
        student_search_results_path: Output of search_dataset.
        save_directory: Directory the answers file is written to.
    """
    from src.oracle import Oracle

    with open(student_search_results_path, "r") as f:
        student_search_results = StudentSearchResults.model_validate_json(
            f.read()
        )
    results = student_search_results.search_results
    print(f"Loaded {len(results)} questions")
    model = Oracle()
    answers = []
    with tqdm(total=len(results), desc="Answering") as progress:
        for i in range(0, len(results), ANSWER_BATCH_SIZE):
            batch = results[i : i + ANSWER_BATCH_SIZE]
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


def evaluate(student_search_results_path: str, dataset_path: str) -> None:
    """Prints recall@k of a StudentSearchResults file against a ground-truth
    AnsweredQuestions dataset.

    Args:
        student_search_results_path: Output of search_dataset.
        dataset_path: Ground-truth AnsweredQuestions dataset.
    """
    evaluator = Evaluator()
    recalls = evaluator.calculate_recall_at_k(
        student_search_results_path, dataset_path
    )
    print("Evaluation Results")
    print("=" * 40)
    print("  ".join(f"{name}: {value:.3f}" for name, value in recalls.items()))
