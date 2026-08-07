import ast
import re
import os
from typing import List, Optional
from pydantic import BaseModel, Field, validate_call
from src.models import MinimalSource
from src.errors import ErrorChunking


class Chunker(BaseModel):
    @validate_call
    def chunk_md(
        self,
        file_path: str,
        content: Optional[str] = None,
        sep_index: int = 0,
        sep_list: List[str] = Field(
            min_lenght=1, default=["\n#", "\n##", "\n###", "\n\n", "\n", " "]
        ),
        max_chunk_size: int = Field(lt=2001, default=2000),
    ) -> List[MinimalSource] | None:

        recursive_chunks = None
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

        for chunk in raw_chunks:
            if len(chunk) < max_chunk_size:
                start_index = content.find(chunk, current_index)
                end_index = start_index + len(chunk)
                chunks.append(
                    MinimalSource(
                        file_path=file_path,
                        first_character_index=start_index,
                        last_character_index=end_index,
                        content=chunk,
                    )
                )
                current_index = end_index

            else:
                if sep_index + 1 < len(sep_list):
                    recursive_chunks = self.chunk_md(
                        file_path=file_path,
                        content=chunk,
                        sep_index=(sep_index + 1),
                    )

                if recursive_chunks is not None:
                    chunks = chunks + recursive_chunks
                else:
                    raise ErrorChunking(file_path)

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

        try:
            tree = ast.parse(content)
        #            print(ast.dump(tree, indent=4))
        #           input()
        except SyntaxError:
            return self.chunk_md(
                file_path=file_path,
                content=content,
                sep_list=["class", "def", "for", "\n", " "],
                max_chunk_size=max_chunk_size,
            )

        for node in tree.body:
            start_line = node.lineno - 1
            end_line = node.end_lineno

            chunk_text = "".join(lines[start_line:end_line])

            if len(chunk_text) < max_chunk_size:
                start_index = sum(len(line) for line in lines[:start_line])
                end_index = start_index + len(chunk_text)

                chunks.append(
                    MinimalSource(
                        file_path=file_path,
                        first_character_index=start_index,
                        last_character_index=end_index,
                        content=chunk_text,
                    )
                )
            else:
                chunks = chunks + self.chunk_md(
                    file_path=file_path,
                    content=content,
                    sep_list=[
                        "class",
                        "def",
                        "for",
                        "\n\n\n",
                        "\n\n",
                        "\n",
                        " ",
                    ],
                    max_chunk_size=max_chunk_size,
                )
                for chunk in chunks:
                    print(chunk)
                    input()

        return chunks

    def dump_index(
        self,
        sources: List[MinimalSource],
        file_path: str = "data/processed/index.jsonl",
    ) -> None:

        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "w") as f:
            for source in sources:
                f.write(source.model_dump_json() + "\n")
