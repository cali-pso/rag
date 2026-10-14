import functools
import os
import sys
from typing import Any, Callable
import fire
from src.errors import RagError
from src.commands import (
    index,
    search,
    search_dataset,
    answer,
    answer_dataset,
    evaluate,
)


def _cli(func: Callable[..., Any]) -> Callable[..., None]:
    """Wraps a command so that no exception ever reaches the user as a
    traceback: each one becomes a one-line message on stderr and a
    non-zero exit status. Set RAG_DEBUG=1 to get the traceback back."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> None:
        try:
            func(*args, **kwargs)
        except KeyboardInterrupt:
            print("\nInterrupted.", file=sys.stderr)
            sys.exit(130)
        except Exception as e:
            if os.environ.get("RAG_DEBUG"):
                raise
            if isinstance(e, (RagError, OSError, ValueError)):
                print(f"Error: {e}", file=sys.stderr)
            else:
                print(
                    f"Unexpected error ({type(e).__name__}): {e}",
                    file=sys.stderr,
                )
            sys.exit(1)

    return wrapper


if __name__ == "__main__":
    fire.Fire({
        "index": _cli(index),
        "search": _cli(search),
        "search_dataset": _cli(search_dataset),
        "answer": _cli(answer),
        "answer_dataset": _cli(answer_dataset),
        "evaluate": _cli(evaluate),
    })
