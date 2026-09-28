import os
from src.models import StudentSearchResults
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_NAME = "Qwen/Qwen3-0.6B"


class Oracle:
    def __init__(self) -> None:
        torch.set_num_threads(os.cpu_count())
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        self.inputs = None

    def build_prompt(self, search_results: StudentSearchResults) -> None:
        self.prompt = ""
        context = search_results.search_results
        question = search_results.question
        messages = [
            {
                "role": "system",
                "content": (
                    "Answer the question using only the provided context. "
                    "If the context doesn't contain the answer, "
                    "say so explicitly."
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            },
        ]

        self.inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            enable_thinking=False,
        )

    def answer(self) -> str:
        assert self.inputs is not None
        
