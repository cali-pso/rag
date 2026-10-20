"""Recall@k computation, for the evaluate command."""
from typing import Dict, List, Tuple
from src.errors import UnknownQuestionID
from src.utils import read_json_model
from src.models import (
    RagDataset,
    AnsweredQuestion,
    MinimalSource,
    StudentSearchResults,
    MinimalSearchResults,
)

K_VALUES = (10, 8, 5, 3, 1)
IOU_THRESHOLD = 0.05


class Evaluator:
    """Computes recall@k of student search results against a
    ground-truth AnsweredQuestions dataset, the way the moulinette does."""

    def __init__(self) -> None:
        """Starts with nothing loaded."""
        self.ground_truth_set = RagDataset(rag_questions=[])
        self.student_answers = StudentSearchResults(search_results=[], k=0)
        self.question_pairs: List[
            Tuple[AnsweredQuestion, MinimalSearchResults]
        ] = []
        # Ground-truth questions left out of the score, and why.
        self.unmatched = 0
        self.without_sources = 0

    def load_ground_truth(self, path: str) -> None:
        """Reads the ground-truth dataset."""
        self.ground_truth_set = read_json_model(path, RagDataset)

    def load_student_answers(self, path: str) -> None:
        """Reads the student search results."""
        self.student_answers = read_json_model(path, StudentSearchResults)

    def build_pairs(self) -> None:
        """Pairs each ground-truth question with the student result that
        has the same question_id.

        Questions without ground-truth sources (an UnansweredQuestions
        file passed by mistake) or without a student result are skipped
        and counted instead of aborting the whole evaluation.

        Raises:
            UnknownQuestionID: If no question can be evaluated at all.
        """
        answers_by_id = {
            i.question_id: i for i in self.student_answers.search_results
        }
        self.question_pairs = []
        self.unmatched = 0
        self.without_sources = 0
        for truth in self.ground_truth_set.rag_questions:
            if not isinstance(truth, AnsweredQuestion):
                self.without_sources += 1
            elif truth.question_id not in answers_by_id:
                self.unmatched += 1
            else:
                self.question_pairs.append(
                    (truth, answers_by_id[truth.question_id])
                )
        if self.question_pairs:
            return
        if not self.ground_truth_set.rag_questions:
            reason = "the ground-truth dataset contains no questions"
        elif not self.unmatched:
            reason = (
                "none of the ground-truth questions has sources: "
                "--dataset_path must be an AnsweredQuestions file"
            )
        else:
            reason = (
                "no question_id in common with the student results (are "
                "both files from the same dataset?)"
            )
        raise UnknownQuestionID(f"nothing to evaluate: {reason}.")

    @staticmethod
    def sources_paths_match(
        answer: MinimalSource, truth: MinimalSource
    ) -> bool:
        """Whether both sources point at the same file."""
        return answer.file_path == truth.file_path

    @staticmethod
    def iou(answer: MinimalSource, truth: MinimalSource) -> float:
        """Intersection over union of the two character ranges."""
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
    ) -> Dict[str, float]:
        """Computes the mean recall@k over every evaluable question.

        Args:
            answer_path: StudentSearchResults JSON file.
            truth_path: Ground-truth AnsweredQuestions JSON file.

        Returns:
            The mean recall for each k in K_VALUES, keyed "recall@k".
        """
        self.load_ground_truth(truth_path)
        self.load_student_answers(answer_path)
        self.build_pairs()
        totals: Dict[int, float] = {k: 0.0 for k in K_VALUES}
        for truth, answer in self.question_pairs:
            if not truth.sources:
                continue
            for k in K_VALUES:
                top_k = answer.retrieved_sources[:k]
                found = sum(
                    1
                    for t_source in truth.sources
                    if any(
                        self.sources_paths_match(a_source, t_source)
                        and self.iou(a_source, t_source) >= IOU_THRESHOLD
                        for a_source in top_k
                    )
                )
                totals[k] += found / len(truth.sources)

        num_questions = len(self.question_pairs)
        return {f"recall@{k}": totals[k] / num_questions for k in K_VALUES}
