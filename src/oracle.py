import os
from typing import List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.models import MinimalSearchResults

MODEL_NAME = "Qwen/Qwen3-0.6B"
MAX_NEW_TOKENS = 256
SYSTEM_PROMPT = (
    "Answer the question using the provided context. If the "
    "context does not relate to the answer, say so explicitly. Be "
    "concise and factual."
)


class Oracle:
    def __init__(
        self,
        model_name: str = MODEL_NAME,
        max_new_tokens: int = MAX_NEW_TOKENS,
    ) -> None:
        # __main__ pins OMP/MKL/OPENBLAS to 1 thread for the OpenVINO
        # embedding model's benefit; that throttles plain PyTorch
        # generation too unless explicitly overridden here.
        torch.set_num_threads(os.cpu_count() or 4)

        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Required for batched causal-LM generation: every row's real
        # content must end at the same position so completions can be
        # sliced out at one fixed offset across the whole batch.
        self.tokenizer.padding_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float32,
            low_cpu_mem_usage=True,
        )
        self.model.eval()

    def _build_conversation(self, result: MinimalSearchResults) -> List[dict]:
        context = "\n\n".join(
            f"[{source.file_path}]\n{source.content}"
            for source in result.retrieved_sources
        )
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Context:\n{context}\n\n"
                    f"Question: {result.question}"
                ),
            },
        ]

    @torch.inference_mode()
    def answer(self, result: MinimalSearchResults) -> str:
        answers: List[str] = self.answer_batch([result])
        return answers[0]

    @torch.inference_mode()
    def answer_batch(self, results: List[MinimalSearchResults]) -> List[str]:
        if not results:
            return []

        conversations = [self._build_conversation(r) for r in results]

        inputs = self.tokenizer.apply_chat_template(
            conversations,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            padding=True,
            return_tensors="pt",
            return_dict=True,
        )

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        # Left-padding means every row's prompt occupies the same
        # padded width, so the generated completion starts at this one
        # fixed offset for the whole batch.
        input_len = inputs["input_ids"].shape[1]
        completions = output_ids[:, input_len:]

        return [
            self.tokenizer.decode(c, skip_special_tokens=True).strip()
            for c in completions
        ]
