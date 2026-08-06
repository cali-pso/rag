from typing import List, Optional
from pydantic import BaseModel, Field
from src.models import MinimalSource
from src.errors import ErrorChunking


class Chunker(BaseModel):
    sep_list: List[str] = Field(
        min_lenght=1, default=["\n#", "\n##", "\n###", "\n\n", "\n", " "]
    )

    def chunk(
        self, file_path: str, content: Optional[str] = None, sep_index: int = 0
    ) -> List[MinimalSource] | None:

        recursive_chunks = None
        chunks = []

        if content is None:
            with open(file_path, "r") as f:
                content = f.read()

        raw_chunks = content.split(self.sep_list[sep_index])

        for chunk in raw_chunks:
            if len(chunk) < 2000:
                start_index = content.find(chunk)
                end_index = start_index + len(chunk)
                chunks.append(
                    MinimalSource(
                        file_path=file_path,
                        first_character_index=start_index,
                        last_character_index=end_index,
                        content=chunk,
                        split_on=self.sep_list[sep_index]
                    )
                )

            else:
                if sep_index + 1 < len(self.sep_list):
                    recursive_chunks = self.chunk(
                        file_path=file_path,
                        content=chunk,
                        sep_index=(sep_index + 1),
                    )

                if recursive_chunks is not None:
                    chunks = chunks + recursive_chunks
                else:
                    raise ErrorChunking(file_path)

        return chunks
