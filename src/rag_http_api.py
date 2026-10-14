from src.commands import index, answer, search
from flask import Flask, request
from flask.typing import ResponseReturnValue

app = Flask("rag_http_api")


@app.route("/index", methods=["POST"])
def http_api_index() -> ResponseReturnValue:
    """Builds the index from the form's max_chunk_size, returns stats."""
    try:
        max_chunk_size = int(request.form["max_chunk_size"])
    except Exception as e:
        return f"Error parsing argument: {e}"
    return index(max_chunk_size)


@app.route("/search", methods=["POST"])
def http_api_search() -> ResponseReturnValue:
    """Returns the top-k sources for the form's query as JSON."""
    try:
        query = str(request.form["query"])
        k = int(request.form["k"])
    except Exception as e:
        return f"Error parsing argument: {e}"
    return search(query=query, k=k)


@app.route("/answer", methods=["POST"])
def http_api_answer() -> ResponseReturnValue:
    """Answers the form's query from its top-k sources, as JSON."""
    try:
        query = str(request.form["query"])
        k = int(request.form["k"])
    except Exception as e:
        return f"Error parsing argument: {e}"
    result = answer(query=query, k=k)
    if result is None:
        return "Error: empty query, there is nothing to answer."
    return result
