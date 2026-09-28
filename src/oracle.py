from src.models import StudentSearchResults


class Oracle:
    def __init__(self) -> None:
        pass

    def build_prompt(self, search_results: StudentSearchResults) -> None:
        self.prompt = ""
