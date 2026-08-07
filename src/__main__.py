import os
import time
import fire
from typing import Dict, List
from src.chunker import Chunker


def sort_files(root_path: str) -> Dict[str, List[str]]:
    md_paths = []
    py_paths = []

    for root, _, files in os.walk(root_path):
        for file in files:
            if file.endswith(".md"):
                md_paths.append(os.path.join(root, file))
            elif file.endswith(".py"):
                py_paths.append(os.path.join(root, file))
    return {"md_paths": md_paths, "py_paths": py_paths}


def index(max_chunk_size: int = 2000) -> None:
    start = time.time()
    paths = sort_files("data")
    chunker = Chunker()
    sources = []
    chunks = 0
    files = 0
    for file in paths["md_paths"]:
        for chunk in chunker.chunk_md(
            file_path=file, max_chunk_size=max_chunk_size
        ):
            chunks += 1
            sources.append(chunk)
        files += 1
    for file in paths["py_paths"]:
        for chunk in chunker.chunk_py(
            file_path=file, max_chunk_size=max_chunk_size
        ):
            chunks += 1
            sources.append(chunk)
        files += 1
    chunker.dump_index(sources)
    print(f"Indexed {files} files as {chunks} sources in {time.time() - start}s")


if __name__ == "__main__":
    fire.Fire()
