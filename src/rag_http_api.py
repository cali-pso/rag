from src.commands import index, answer, search, _build_retriever
import threading
from flask import Flask, request

app = Flask("rag_http_api")

try:
    _build_retriever()
except Exception:
    pass  # no index yet: the first request will report it



@app.route("/index", methods=["POST"])
def http_api_index():
    try:
        max_chunk_size = int(request.form["max_chunk_size"])
    except Exception as e:
        return f"Error parsing argument: {e}"
    return index(max_chunk_size)


@app.route("/search", methods=["POST"])
def http_api_search():
    try:
        query = str(request.form["query"])
        k = int(request.form["k"])
    except Exception as e:
        return f"Error parsing argument: {e}"
    return search(query=query, k=k, silent=True)


@app.route("/answer", methods=["POST"])
def http_api_answer():
    try:
        query = str(request.form["query"])
        k = int(request.form["k"])
    except Exception as e:
        return f"Error parsing argument: {e}"
    return answer(query=query, k=k)
