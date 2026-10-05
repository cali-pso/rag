import fire
from src.commands import (
    index,
    search,
    search_dataset,
    answer,
    answer_dataset,
    evaluate,
)

if __name__ == "__main__":
    fire.Fire({
        "index": index,
        "search": search,
        "search_dataset": search_dataset,
        "answer": answer,
        "answer_dataset": answer_dataset,
        "evaluate": evaluate,
    })
