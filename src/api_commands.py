from src.commands import index
from flask import Flask, request

app = Flask("rag_http_api")


@app.route("/index", methods=["POST"])
def http_api_index():
    return index(int(request.form["max_chunk_size"]))
