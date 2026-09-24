from typing import Dict
from src.errors import UnknownQuestionID
from src.models import (
    RagDataset,
    AnsweredQuestion,
    MinimalSource,
    StudentSearchResults,
    MinimalSearchResults,
)


class Evaluator:
    def __init__(self) -> None:
        pass

    def load_ground_truth(self, path: str) -> None:
        with open(path, "r") as f:
            self.ground_truth_set = RagDataset.model_validate_json(f.read())

    def load_student_answers(self, path: str) -> None:
        with open(path, "r") as f:
            self.student_answers = StudentSearchResults.model_validate_json(
                f.read()
            )

    def build_pairs(self) -> None:
        self.question_pairs: Dict[
            str, tuple[AnsweredQuestion, MinimalSearchResults]
        ] = {}
        answers_by_id = {
            i.question_id: i for i in self.student_answers.search_results
        }
        for truth in self.ground_truth_set.rag_questions:
            if truth.question_id not in answers_by_id:
                raise UnknownQuestionID(
                    f"Question {truth.question_id} from ground truth "
                    "set has no match in student answers"
                )
            self.question_pairs[truth.question_id] = (
                truth,
                answers_by_id[truth.question_id],
            )

    @staticmethod
    def sources_paths_match(
        answer: MinimalSource, truth: MinimalSource
    ) -> bool:
        return answer.file_path == truth.file_path

    @staticmethod
    def iou(answer: MinimalSource, truth: MinimalSource) -> float:
        i_start = max(
            answer.first_character_index, truth.first_character_index
        )
        i_end = min(answer.last_character_index, truth.last_character_index)
        i_len = max(0, i_end - i_start)

        answer_len = answer.last_character_index - answer.first_character_index
        truth_len = truth.last_character_index - truth.first_character_index
        union_len = answer_len + truth_len - i_len

        return i_len / union_len if union_len > 0 else 0.0

    def calculate_recall_at_k(
        self, answer_path: str, truth_path: str
    ) -> Dict[str, int]:
        self.load_ground_truth(truth_path)
        self.load_student_answers(answer_path)
        self.build_pairs()
        full_results: Dict[str, Dict[int, float]] = {}
        for truth, answer in self.question_pairs.values():
            q_results: Dict[int, float] = {}
            for k in (10, 5, 1):
                top_k = answer.retrieved_sources[:k]
                found = 0
                for t_source in truth.sources:
                    if any(
                        self.sources_paths_match(a_source, t_source)
                        and self.iou(a_source, t_source) >= 0.05
                        for a_source in top_k
                    ):
                        found += 1
                q_results[k] = (
                    found / len(truth.sources) if truth.sources else 0.0
                )
            full_results[truth.question_id] = q_results

        k_values = next(iter(full_results.values())).keys()
        num_questions = len(full_results)

        return {
            f"recall@{k}": sum(q[k] for q in full_results.values())
            / num_questions
            for k in k_values
        }
