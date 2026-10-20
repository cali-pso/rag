from src.commands import index, answer, search_dataset, search, get_config, set_config, _build_retriever
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

@app.route("/search_dataset", methods=["POST"])
def http_api_search_dataset():
    try:
        dataset_path = str(request.form["dataset_path"])
        k = int(request.form["k"])
    except Exception as e:
        return f"Error parsing argument: {e}"
    return search_dataset(dataset_path=dataset_path, k=k)


@app.route("/answer", methods=["POST"])
def http_api_answer():
    try:
        query = str(request.form["query"])
        k = int(request.form["k"])
    except Exception as e:
        return f"Error parsing argument: {e}"
    return answer(query=query, k=k)

@app.route("/ping", methods=["GET", "POST"])
def ping():
    return "pong\n"

@app.route("/set_config", methods=["POST"])
def http_api_set_config():
    args = request.form.to_dict()
    try:
        return set_config(**args)
    except Exception as e:
        return f"Error parsing argument: {e}"

@app.route("/get_config", methods=["GET"])
def http_api_get_config():
    return get_config()