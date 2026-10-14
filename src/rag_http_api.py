import threading
from typing import Any, Callable, Mapping, Optional
from flask import Flask, request
from flask.typing import ResponseReturnValue
from src.commands import index, answer, search
from src.errors import RagError

app = Flask(__name__)
# Requests share the models cached in memory and /index rewrites the
# files they read: serve one request at a time.
_LOCK = threading.Lock()


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
    """Runs a command and turns its result, or its failure, into a JSON
    response: 400 for a rejected input, 500 for anything unexpected."""
    try:
        with _LOCK:
            result = command(**kwargs)
    except (RagError, ValueError, OSError) as e:
        return {"error": str(e)}, 400
    except Exception as e:
        return {"error": f"unexpected error ({type(e).__name__}): {e}"}, 500
    if result is None:
        return {"error": "empty query, there is nothing to answer"}, 400
    return result


@app.errorhandler(BadRequest)
def _bad_request(e: BadRequest) -> ResponseReturnValue:
    """Answers a malformed request with a JSON error."""
    return {"error": str(e)}, 400


@app.route("/index", methods=["POST"])
def http_api_index() -> ResponseReturnValue:
    """Builds the index from the form's max_chunk_size, returns stats."""
    return _run(index, max_chunk_size=_form_int("max_chunk_size"))


@app.route("/search", methods=["POST"])
def http_api_search() -> ResponseReturnValue:
    """Returns the top-k sources for the form's query as JSON."""
    return _run(search, query=_form_str("query"), k=_form_int("k"))


@app.route("/answer", methods=["POST"])
def http_api_answer() -> ResponseReturnValue:
    """Answers the form's query from its top-k sources, as JSON."""
    return _run(answer, query=_form_str("query"), k=_form_int("k"))
