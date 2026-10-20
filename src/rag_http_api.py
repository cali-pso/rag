"""Local HTTP API over the CLI commands (bonus)."""
import threading
from typing import Any, Callable, Mapping, Optional
from flask import Flask, request
from flask.typing import ResponseReturnValue
from src.commands import (
    _build_retriever,
    answer,
    index,
    search,
    search_dataset,
)
from src.errors import RagError

app = Flask("rag_http_api")
# Requests share the retriever and model cached in memory, and /index
# rewrites the files they read: serve one request at a time.
_LOCK = threading.Lock()

try:
    _build_retriever()  # loads the index now: the first search is fast too
except RagError:
    pass  # no index yet: the first request will report it


class BadRequest(Exception):
    """A form field is missing or malformed."""


def _form_str(name: str) -> str:
    """Returns a required form field."""
    value = request.form.get(name)
    if value is None:
        raise BadRequest(f"missing form field '{name}'")
    return value


def _form_int(name: str) -> int:
    """Returns a required integer form field."""
    value = _form_str(name)
    try:
        return int(value)
    except ValueError:
        raise BadRequest(f"'{name}' must be an integer, got {value!r}")


def _run(
    command: Callable[..., Optional[Mapping[str, Any]]], **kwargs: Any
) -> ResponseReturnValue:
    """Runs a command and returns its result, or its error, as JSON."""
    try:
        with _LOCK:
            result = command(**kwargs)
    except (RagError, ValueError, OSError) as e:
        return {"error": str(e)}, 400
    except Exception as e:
        return {"error": f"unexpected error ({type(e).__name__}): {e}"}, 500
    if result is None:
        return {"error": "empty query, there is nothing to answer"}, 400
    return dict(result)


@app.errorhandler(BadRequest)
def _bad_request(e: BadRequest) -> ResponseReturnValue:
    """Answers a malformed request with a JSON error."""
    return {"error": str(e)}, 400


@app.route("/index", methods=["POST"])
def http_api_index() -> ResponseReturnValue:
    """Builds the index with the form's max_chunk_size."""
    return _run(index, max_chunk_size=_form_int("max_chunk_size"))


@app.route("/search", methods=["POST"])
def http_api_search() -> ResponseReturnValue:
    """Returns the top-k sources for the form's query."""
    return _run(search, query=_form_str("query"), k=_form_int("k"),
                silent=True)


@app.route("/search_dataset", methods=["POST"])
def http_api_search_dataset() -> ResponseReturnValue:
    """Searches every question of the form's dataset."""
    return _run(
        search_dataset,
        dataset_path=_form_str("dataset_path"),
        k=_form_int("k"),
    )


@app.route("/answer", methods=["POST"])
def http_api_answer() -> ResponseReturnValue:
    """Answers the form's query from its top-k sources."""
    return _run(answer, query=_form_str("query"), k=_form_int("k"))


@app.route("/ping", methods=["GET", "POST"])
def ping() -> str:
    """Health check."""
    return "pong\n"
