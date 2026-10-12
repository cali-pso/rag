from src.commands import index, answer
from flask import Flask, request

app = Flask("rag_http_api")


@app.route("/index", methods=["POST"])
def http_api_index():
    return index(int(request.form["max_chunk_size"]))


@app.route("/answer", methods=["POST"])
def http_api_answer():
    return answer(query=str(request.form["query"]), k=int(request.form["k"]))
