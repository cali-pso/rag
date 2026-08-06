from src.chunker import Chunker


def test_main() -> None:
    chunker = Chunker()
    chunks = chunker.chunk("README.md")
    for chunk in chunks:
        print(f"{chunk.content}\n---------- {chunk.file_path} - {chunk.first_character_index} - {chunk.last_character_index} - {chunk.split_on.replace("\n", "n")} ------------\n")


if __name__ == "__main__":
    test_main()
