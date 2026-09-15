import ast
import re
import os
from typing import List, Optional, Dict, Set
from pydantic import BaseModel, Field, validate_call
from src.models import MinimalSource
from src.errors import ErrorChunking

LEXICAL_INDEX_PATH = "data/processed/lexical_index.json"


class LexicalIndexer(BaseModel):
    chunks: Dict[str, List[MinimalSource]] = {}

    @validate_call
    def create_index(
            self, paths: Dict[str, List[str]], deleted_paths: Set[str], max_chunk_size: int = 2000
    ) -> Dict[str, List[MinimalSource]]:
        files = 0
        self.chunks = load_index(deleted_paths)
        for file in paths["md_paths"]:
            self.chunks[file] = []
            for chunk in self.chunk_md(
                file_path=file, max_chunk_size=max_chunk_size
            ):
                self.chunks[file].append(chunk)
            files += 1
        for file in paths["py_paths"]:
            self.chunks[file] = []
            for chunk in self.chunk_py(
                file_path=file, max_chunk_size=max_chunk_size
            ):
                self.chunks[file].append(chunk)
            files += 1
        self.dump_index()

    @validate_call
    def chunk_md(
        self,
        file_path: str,
        content: Optional[str] = None,
        sep_index: int = 0,
        sep_list: List[str] = Field(
            min_length=1,
            default=["\n# ", "\n## ", "\n### ", "\n\n", "\n", " "],
        ),
        max_chunk_size: int = Field(lt=2001, default=2000),
        offset: int = 0,
    ) -> List[MinimalSource]:

        chunks = []
        current_index = 0

        if content is None:
            with open(file_path, "r") as f:
                content = f.read()

        escaped_sep = re.escape(sep_list[sep_index])

        raw_split = re.split(f"({escaped_sep})", content)

        raw_chunks = []
        if raw_split[0]:
            raw_chunks.append(raw_split[0])

        for i in range(1, len(raw_split), 2):
            separator = raw_split[i]
            text_block = raw_split[i + 1]
            raw_chunks.append(separator + text_block)

        buffer_content = ""
        buffer_start = -1

        for chunk in raw_chunks:
            if len(chunk) >= max_chunk_size:
                if buffer_content:
                    chunks.append(
                        MinimalSource(
                            file_path=file_path,
                            first_character_index=buffer_start,
                            last_character_index=buffer_start + len(buffer_content),
                            content=buffer_content,
                        )
                    )
                    buffer_content = ""
                    buffer_start = -1
                recursive_chunks = None
                if sep_index + 1 < len(sep_list):
                    recursive_chunks = self.chunk_md(
                        file_path=file_path,
                        content=chunk,
                        sep_index=(sep_index + 1),
                        sep_list=sep_list,
                        max_chunk_size=max_chunk_size,
                        offset=offset + current_index,
                    )
                if recursive_chunks is not None:
                    chunks = chunks + recursive_chunks
                else:
                    raise ErrorChunking(file_path)
            elif len(buffer_content) + len(chunk) < max_chunk_size:
                if buffer_start == -1:
                    buffer_start = offset + current_index
                buffer_content += chunk
            else:
                chunks.append(
                    MinimalSource(
                        file_path=file_path,
                        first_character_index=buffer_start,
                        last_character_index=buffer_start + len(buffer_content),
                        content=buffer_content,
                    )
                )
                buffer_content = chunk
                buffer_start = offset + current_index
            current_index += len(chunk)
        if buffer_content:
            chunks.append(
                MinimalSource(
                    file_path=file_path,
                    first_character_index=buffer_start,
                    last_character_index=buffer_start + len(buffer_content),
                    content=buffer_content,
                )
            )

        return chunks

    @validate_call
    def chunk_py(
        self,
        file_path: str,
        content: Optional[str] = None,
        max_chunk_size: int = Field(lt=2001, default=2000),
    ) -> List[MinimalSource]:

        chunks = []
        if content is None:
            with open(file_path, "r") as f:
                content = f.read()

        lines = content.splitlines(keepends=True)
        fallback_sep_list = [
            "\nclass ",
            "\ndef ",
            "\nfor ",
            "\n\n\n",
            "\n\n",
            "\n",
            " ",
        ]

        try:
            tree = ast.parse(content)
        except SyntaxError:
            return self.chunk_md(
                file_path=file_path,
                content=content,
                sep_list=fallback_sep_list,
                max_chunk_size=max_chunk_size,
            )

        line_offsets = [0]
        for line in lines:
            line_offsets.append(line_offsets[-1] + len(line))

        buffer_content = ""
        buffer_start = -1
        buffer_end = 0

        for node in tree.body:
            start_line = node.lineno - 1
            end_line = node.end_lineno
            assert end_line is not None
            node_start = line_offsets[start_line]
            node_end = line_offsets[end_line]

            node_text = "".join(lines[start_line:end_line])
            if len(node_text) > max_chunk_size:
                if buffer_content:
                    chunks.append(
                        MinimalSource(
                            file_path=file_path,
                            first_character_index=buffer_start,
                            last_character_index=buffer_end,
                            content=buffer_content,
                        )
                    )
                    buffer_content = ""
                    buffer_start = -1
                chunks = chunks + self.chunk_md(
                    file_path=file_path,
                    content=node_text,
                    sep_list=fallback_sep_list,
                    max_chunk_size=max_chunk_size,
                    offset=node_start,
                )
            elif len(buffer_content) + len(node_text) < max_chunk_size:
                if buffer_start == -1:
                    buffer_start = node_start
                buffer_content += node_text
                buffer_end = node_end
            else:
                chunks.append(
                    MinimalSource(
                        file_path=file_path,
                        first_character_index=buffer_start,
                        last_character_index=buffer_end,
                        content=buffer_content,
                    )
                )
                buffer_content = node_text
                buffer_start = node_start
                buffer_end = node_end
        if buffer_content:
            chunks.append(
                MinimalSource(
                    file_path=file_path,
                    first_character_index=buffer_start,
                    last_character_index=buffer_end,
                    content=buffer_content,
                )
            )
        return chunks

    def dump_index(
        self,
        file_path: str = LEXICAL_INDEX_PATH,
    ) -> None:

        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "w") as f:
            f.write(self.model_dump_json())


def load_index(
        deleted_paths: Set[str], file_path: str = LEXICAL_INDEX_PATH,
) -> Optional[Dict[str, List[MinimalSource]]]:
    try:
        with open(file_path, "r") as f:
            content = LexicalIndexer.model_validate_json(f.read())
            return {k: v for k, v in content.chunks.items() if k not in deleted_paths}
    except FileNotFoundError:
        return {}
