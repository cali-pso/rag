import ast
import re
import os
from typing import List, Optional, Dict, Set
from pydantic import BaseModel, Field, validate_call
from src.models import MinimalSource
from src.errors import ErrorChunking

LEXICAL_INDEX_PATH = "data/processed/lexical_index.json"
PY_FALLBACK_SEP_LIST = [
    "\nclass ",
    "\ndef ",
    "\nfor ",
    "\n\n\n",
    "\n\n",
    "\n",
    " ",
]


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

        if content is None:
            with open(file_path, "r") as f:
                content = f.read()

        lines = content.splitlines(keepends=True)

        try:
            tree = ast.parse(content)
        except SyntaxError:
            return self.chunk_md(
                file_path=file_path,
                content=content,
                sep_list=PY_FALLBACK_SEP_LIST,
                max_chunk_size=max_chunk_size,
            )

        line_offsets = [0]
        for line in lines:
            line_offsets.append(line_offsets[-1] + len(line))

        return self._chunk_ast_nodes(
            tree.body, lines, line_offsets, file_path, max_chunk_size, 0
        )

    def _chunk_ast_nodes(
        self,
        nodes: List[ast.stmt],
        lines: List[str],
        line_offsets: List[int],
        file_path: str,
        max_chunk_size: int,
        region_start_line: int,
    ) -> List[MinimalSource]:
        """Chunks a run of sibling AST statements, covering every character
        from region_start_line to the last node's end with no gaps: any
        blank lines, comments or decorators preceding a node are folded
        into that node's own text instead of being dropped.

        An oversized class/function recurses into its own body (accurate
        line-based offsets, no risk of splitting mid-identifier); any
        other oversized node falls back to text-based splitting.
        """
        chunks = []
        buffer_content = ""
        buffer_start = -1
        buffer_end = 0
        cursor_line = region_start_line

        for node in nodes:
            end_line = node.end_lineno
            assert end_line is not None
            node_region_start_line = cursor_line
            node_start = line_offsets[node_region_start_line]
            node_end = line_offsets[end_line]
            node_text = "".join(lines[node_region_start_line:end_line])
            cursor_line = end_line

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
                nested_body = (
                    node.body
                    if isinstance(
                        node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
                    )
                    else None
                )
                if nested_body:
                    chunks = chunks + self._chunk_ast_nodes(
                        nested_body,
                        lines,
                        line_offsets,
                        file_path,
                        max_chunk_size,
                        node_region_start_line,
                    )
                else:
                    chunks = chunks + self.chunk_md(
                        file_path=file_path,
                        content=node_text,
                        sep_list=PY_FALLBACK_SEP_LIST,
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
