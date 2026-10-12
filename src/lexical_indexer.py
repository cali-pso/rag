import ast
import re
import os
from typing import List, Optional, Dict, Set
from pydantic import BaseModel, Field, ValidationError, validate_call
from src.models import MinimalSource, make_source_id
from src.errors import CorruptedIndex, ErrorChunking

LEXICAL_INDEX_PATH = "data/processed/lexical_index.json"
# Characters repeated at the start of a new chunk from the tail of the
# previous one. A reference answer that straddles a chunk boundary would
# otherwise miss both neighbors; a small overlap gives it a chance to
# clear the grader's IoU bar in whichever chunk it lands closer to.
CHUNK_OVERLAP = 150
PY_FALLBACK_SEP_LIST = [
    "\nclass ",
    "\ndef ",
    "\nfor ",
    "\n\n\n",
    "\n\n",
    "\n",
    " ",
]


def _snap_forward_to_word_start(text: str, pos: int, limit: int) -> int:
    """Advances pos to the next word-start boundary (preceded by
    whitespace) at or before limit, so an overlap window never begins
    mid-identifier. Only ever moves forward (shrinking the overlap),
    so it can't push a chunk over its size budget. Falls back to
    `limit` (zero overlap for this transition) if no boundary exists
    anywhere in the window."""
    while pos < limit and not (pos == 0 or text[pos - 1].isspace()):
        pos += 1
    return pos


class LexicalIndexer(BaseModel):
    chunks: Dict[str, List[MinimalSource]] = {}

    @validate_call
    def create_index(
        self,
        paths: Dict[str, List[str]],
        deleted_paths: Set[str],
        max_chunk_size: int = 2000,
    ) -> None:
        """Chunks every new or modified file, merges the result with the
        chunks of unchanged files from the previous run and saves it.

        Args:
            paths: New or modified files, under "md_paths" and "py_paths".
            deleted_paths: Previously indexed files that no longer exist.
            max_chunk_size: Maximum width, in characters, of a chunk.
        """
        self.chunks = load_index(deleted_paths)
        jobs = [(file, self.chunk_md) for file in paths["md_paths"]] + [
            (file, self.chunk_py) for file in paths["py_paths"]
        ]
        for file, chunker in jobs:
            try:
                file_chunks = chunker(
                    file_path=file, max_chunk_size=max_chunk_size
                )
            except UnicodeDecodeError:
                # Character offsets of a lossy decode would point at the
                # wrong text, so leave the file out rather than guess.
                print(f"Warning: skipping {file}: not valid UTF-8.")
                self.chunks.pop(file, None)
                continue
            for chunk in file_chunks:
                chunk.faiss_custom_id = make_source_id(
                    chunk.file_path,
                    chunk.first_character_index,
                    chunk.last_character_index,
                )
            self.chunks[file] = file_chunks
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
            with open(file_path, "r", encoding="utf-8") as f:
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
                            last_character_index=buffer_start
                            + len(buffer_content),
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
                else:
                    # No separator left (e.g. a long URL or base64 blob):
                    # cut it into fixed-width slices so an oversized token
                    # never aborts the whole indexing run.
                    step = max_chunk_size - 1
                    recursive_chunks = [
                        MinimalSource(
                            file_path=file_path,
                            first_character_index=offset + current_index + i,
                            last_character_index=offset
                            + current_index
                            + i
                            + len(chunk[i:i + step]),
                            content=chunk[i:i + step],
                        )
                        for i in range(0, len(chunk), step)
                    ]
                if not recursive_chunks:
                    raise ErrorChunking(
                        f"could not split {file_path} into chunks"
                    )
                chunks = chunks + recursive_chunks
            elif len(buffer_content) + len(chunk) < max_chunk_size:
                if buffer_start == -1:
                    buffer_start = offset + current_index
                buffer_content += chunk
            else:
                finished_local_start = buffer_start - offset
                chunks.append(
                    MinimalSource(
                        file_path=file_path,
                        first_character_index=buffer_start,
                        last_character_index=buffer_start
                        + len(buffer_content),
                        content=buffer_content,
                    )
                )
                new_local_start = current_index
                effective_overlap = min(
                    CHUNK_OVERLAP, max_chunk_size - len(chunk)
                )
                overlap_local_start = max(
                    finished_local_start, new_local_start - effective_overlap
                )
                overlap_local_start = _snap_forward_to_word_start(
                    content, overlap_local_start, new_local_start
                )
                buffer_content = (
                    content[overlap_local_start:new_local_start] + chunk
                )
                buffer_start = offset + overlap_local_start
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
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

        lines = content.splitlines(keepends=True)

        try:
            tree = ast.parse(content)
        except (SyntaxError, ValueError, RecursionError):
            # Invalid Python, NUL bytes or absurdly deep nesting: the
            # file can still be indexed as plain text.
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
            tree.body,
            lines,
            line_offsets,
            file_path,
            max_chunk_size,
            0,
            content,
        )

    def _chunk_ast_nodes(
        self,
        nodes: List[ast.stmt],
        lines: List[str],
        line_offsets: List[int],
        file_path: str,
        max_chunk_size: int,
        region_start_line: int,
        content: str,
        prefix: str = "",
    ) -> List[MinimalSource]:
        """Chunks a run of sibling AST statements, covering every character
        from region_start_line to the last node's end with no gaps: any
        blank lines, comments or decorators preceding a node are folded
        into that node's own text instead of being dropped.

        An oversized class/function recurses into its own body (accurate
        line-based offsets, no risk of splitting mid-identifier); any
        other oversized node falls back to text-based splitting. `prefix`
        carries the enclosing class/function signature line(s) so a
        chunk deep inside a class still names the class it belongs to,
        instead of only the first descendant chunk getting that context.
        """
        chunks = []
        buffer_content = ""
        buffer_start = -1
        buffer_end = 0
        cursor_line = region_start_line
        effective_max = max(1, max_chunk_size - len(prefix))

        for node in nodes:
            end_line = node.end_lineno
            assert end_line is not None
            node_region_start_line = cursor_line
            node_start = line_offsets[node_region_start_line]
            node_end = line_offsets[end_line]
            node_text = "".join(lines[node_region_start_line:end_line])
            cursor_line = end_line

            if len(node_text) > effective_max:
                if buffer_content:
                    chunks.append(
                        MinimalSource(
                            file_path=file_path,
                            first_character_index=buffer_start,
                            last_character_index=buffer_end,
                            content=prefix + buffer_content,
                        )
                    )
                    buffer_content = ""
                    buffer_start = -1
                nested_body = (
                    node.body
                    if isinstance(
                        node,
                        (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
                    )
                    else None
                )
                if nested_body:
                    # Everything from this node's own region start (any
                    # gap inherited from its parent, plus its full
                    # signature, however many lines) up to its first body
                    # statement becomes the header text for its children.
                    # The recursive call's own region then starts exactly
                    # at that first body statement, so this text is never
                    # also captured a second time by its cursor-sweep.
                    body_start_line = nested_body[0].lineno - 1
                    header_text = "".join(
                        lines[node_region_start_line:body_start_line]
                    )
                    new_prefix = prefix + header_text
                    # Cap how much of the budget prefix can consume (e.g.
                    # a long decorator + multi-line signature at every
                    # nesting level) so effective_max below never
                    # collapses to near-zero. Keep the tail: the
                    # innermost, most specific context matters most.
                    max_prefix_len = max_chunk_size // 4
                    if len(new_prefix) > max_prefix_len:
                        cut = _snap_forward_to_word_start(
                            new_prefix,
                            len(new_prefix) - max_prefix_len,
                            len(new_prefix),
                        )
                        new_prefix = new_prefix[cut:]
                    chunks = chunks + self._chunk_ast_nodes(
                        nested_body,
                        lines,
                        line_offsets,
                        file_path,
                        max_chunk_size,
                        body_start_line,
                        content,
                        new_prefix,
                    )
                else:
                    fallback_chunks = self.chunk_md(
                        file_path=file_path,
                        content=node_text,
                        sep_list=PY_FALLBACK_SEP_LIST,
                        max_chunk_size=effective_max,
                        offset=node_start,
                    )
                    if prefix:
                        fallback_chunks = [
                            MinimalSource(
                                file_path=fc.file_path,
                                first_character_index=fc.first_character_index,
                                last_character_index=fc.last_character_index,
                                content=prefix + fc.content,
                            )
                            for fc in fallback_chunks
                        ]
                    chunks = chunks + fallback_chunks
            elif len(buffer_content) + len(node_text) < effective_max:
                if buffer_start == -1:
                    buffer_start = node_start
                buffer_content += node_text
                buffer_end = node_end
            else:
                # A node exactly effective_max long lands here with an
                # empty buffer (buffer_start still -1): nothing to flush.
                if buffer_content:
                    chunks.append(
                        MinimalSource(
                            file_path=file_path,
                            first_character_index=buffer_start,
                            last_character_index=buffer_end,
                            content=prefix + buffer_content,
                        )
                    )
                effective_overlap = min(
                    CHUNK_OVERLAP, effective_max - len(node_text)
                )
                overlap_start = max(
                    buffer_start, node_start - effective_overlap
                )
                overlap_start = _snap_forward_to_word_start(
                    content, overlap_start, node_start
                )
                buffer_content = content[overlap_start:node_start] + node_text
                buffer_start = overlap_start
                buffer_end = node_end
        if buffer_content:
            chunks.append(
                MinimalSource(
                    file_path=file_path,
                    first_character_index=buffer_start,
                    last_character_index=buffer_end,
                    content=prefix + buffer_content,
                )
            )
        return chunks

    def dump_index(
        self,
        file_path: str = LEXICAL_INDEX_PATH,
    ) -> None:

        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(self.model_dump_json())


def load_index(
    deleted_paths: Set[str],
    file_path: str = LEXICAL_INDEX_PATH,
) -> Dict[str, List[MinimalSource]]:
    """Reads the chunks saved by the previous run, minus deleted files.

    Args:
        deleted_paths: Files to leave out of the result.
        file_path: Where the lexical index is saved.

    Returns:
        The chunks of each file, or an empty dict if nothing was saved.

    Raises:
        CorruptedIndex: If the saved index cannot be parsed.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = LexicalIndexer.model_validate_json(f.read())
    except FileNotFoundError:
        return {}
    except (ValidationError, UnicodeDecodeError):
        raise CorruptedIndex(
            f"{file_path} is corrupted, run `index` again to rebuild it."
        )
    return {
        k: v for k, v in content.chunks.items() if k not in deleted_paths
    }
