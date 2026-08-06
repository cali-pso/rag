import os
import time
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


def test_main() -> None:
    start = time.time()
    paths = sort_files("data")
    chunker = Chunker()
    files = 0
    for file in paths["md_paths"]:
        chunker.chunk_md(file_path=file)
        files += 1
    for file in paths["py_paths"]:
        chunker.chunk_py(file_path=file)
        files += 1
    print(f"Indexed {files} files in {time.time() - start}s")


if __name__ == "__main__":
    test_main()
