from typing import Dict
from src.errors import UnknownQuestionID
from src.models import RagDataset, AnsweredQuestion, MinimalSource


class Evaluator:
    def __init__(self) -> None:
        pass

    def load_ground_truth(self, path: str) -> None:
        with open(path, "r") as f:
            self.ground_truth_set = RagDataset.model_validate_json(f.read())
        for truth in self.ground_truth_set.rag_questions:
            assert isinstance(truth, AnsweredQuestion)

    def load_student_answers(self, path: str) -> None:
        with open(path, "r") as f:
            self.student_answers_set = RagDataset.model_validate_json(f.read())
        for answer in self.student_answers_set.rag_questions:
            assert isinstance(answer, AnsweredQuestion)

    def build_pairs(self) -> None:
        self.question_pairs: Dict[AnsweredQuestion, AnsweredQuestion] = dict()
        answers_ids = {i.question_id: i for i in self.student_answers_set}
        for truth in self.ground_truh_set.rag_questions:
            if truth.question_id not in answers_ids.keys():
                raise UnknownQuestionID(
                    f"Question {truth.question_id} from ground truth",
                    "set has no match in student answers",
                )
            self.question_pairs[truth] = answers_ids[truth.question_id]

    def sources_paths_match(
        answer: MinimalSource, truth: MinimalSource
    ) -> bool:
        return answer.file_path == truth.file_path
