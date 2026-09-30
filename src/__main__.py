import os
import time
import fire
from src.lexical_indexer import load_index
from src.bm25retriever import BM25Retriever
from src.vector_indexer import VectorIndexer
from src.hybrid_retriever import HybridRetriever
from src.evaluator import Evaluator
from src.oracle import Oracle
from src.utils import load_cache, save_cache, sort_files, load_dataset
from src.models import (
    UnansweredQuestion,
    MinimalSearchResults,
    StudentSearchResults,
    MinimalAnswer,
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
    lexical_retriever = BM25Retriever()
    cache = load_cache()
    _, deleted_paths = sort_files("data/raw", cache)
    sources_dict = load_index(deleted_paths)
    sources = []
    for file in sources_dict:
        for source in sources_dict[file]:
            sources.append(source)
    lexical_retriever.fit(sources)
    semantic_retriever = VectorIndexer()
    semantic_retriever.load()
    hybrid_retriever = HybridRetriever(lexical_retriever, semantic_retriever)
    answer = hybrid_retriever.search(question.question, k)
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


def search_dataset(
    dataset_path: str,
    save_directory: str,
    k: int = 10,
):
    from src.vector_indexer import VectorIndexer
    from src.hybrid_retriever import HybridRetriever

    dataset = load_dataset(dataset_path)
    bm25 = BM25Retriever()
    cache = load_cache()
    _, deleted_paths = sort_files("data/raw", cache)
    sources_dict = load_index(deleted_paths)
    sources = []
    for file in sources_dict:
        for source in sources_dict[file]:
            sources.append(source)
    bm25.fit(sources)

    vector_indexer = VectorIndexer()
    retriever = HybridRetriever(bm25=bm25, vector_indexer=vector_indexer)

    print(f"Treating {len(dataset.rag_questions)}")
    queries = [question.question for question in dataset.rag_questions]
    batch_results = retriever.search_batch(queries, k)

    search_results = [
        MinimalSearchResults(
            question_id=question.question_id,
            question=question.question,
            retrieved_sources=answer,
        )
        for question, answer in zip(dataset.rag_questions, batch_results)
    ]
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


def answer(student_search_results_path: str, save_directory: str) -> None:
    with open(student_search_results_path, "r") as f:
        student_search_results = StudentSearchResults.model_validate_json(
            f.read()
        )
    model = Oracle()
    result = student_search_results.search_results[0]
    answer = MinimalAnswer(
        answer=model.answer(student_search_results.search_results[0]),
        question_id=result.question_id,
        question=result.question,
        retrieved_sources=result.retrieved_sources,
    )
    print(answer.answer)
    pass


def answer_dataset(
    student_search_results_path: str, save_directory: str
) -> None:
    with open(student_search_results_path, "r") as f:
        student_search_results = StudentSearchResults.model_validate_json(
            f.read()
        )
    model = Oracle()
    answer_batch = model.answer_batch(student_search_results.search_results)
    print(answer_batch)
    pass


def evaluate_student_search_results(answer_path: str, truth_path: str) -> None:
    evaluator = Evaluator()
    print(evaluator.calculate_recall_at_k(answer_path, truth_path))


if __name__ == "__main__":
    fire.Fire()
