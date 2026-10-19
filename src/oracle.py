from pathlib import Path
from typing import Dict, List

import openvino_genai as og
from src.models import MinimalSearchResults
from src.vector_indexer import OV_CACHE_DIR


MODEL_NAME = "Qwen/Qwen3-0.6B"
QWEN_OV_DIR = Path("data/models/qwen3-0.6b-ov-int8")
MAX_NEW_TOKENS = 128
SYSTEM_PROMPT = (
    "Answer the question using the provided context. If the "
    "context does not relate to the answer, say so explicitly. Be "
    "concise and factual, answer in 2-3 phrases."
)
SOURCES_INCLUDED = 3


def load_qwen() -> og.LLMPipeline:
    """Exports Qwen3-0.6B to OpenVINO with int8 weights on first use, then
    loads it as an OpenVINO GenAI pipeline."""
    from optimum.intel import OVModelForCausalLM, OVWeightQuantizationConfig
    from transformers import AutoTokenizer
    import openvino as ov
    from openvino_tokenizers import convert_tokenizer


    if not (QWEN_OV_DIR / "openvino_model.xml").exists():
        model = OVModelForCausalLM.from_pretrained(
            MODEL_NAME,
            export=True,
            quantization_config=OVWeightQuantizationConfig(bits=8),
        )
        model.save_pretrained(QWEN_OV_DIR)
    # LLMPipeline needs the tokenizer in OpenVINO format too, which
    # OVModelForCausalLM.save_pretrained() does not write.
    if not (QWEN_OV_DIR / "openvino_tokenizer.xml").exists():
        hf_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        ov_tokenizer, ov_detokenizer = convert_tokenizer(
            hf_tokenizer, with_detokenizer=True
        )
        ov.save_model(ov_tokenizer, QWEN_OV_DIR / "openvino_tokenizer.xml")
        ov.save_model(
            ov_detokenizer, QWEN_OV_DIR / "openvino_detokenizer.xml"
        )
        hf_tokenizer.save_pretrained(QWEN_OV_DIR)
    return og.LLMPipeline(str(QWEN_OV_DIR), "CPU")


class Oracle:
    def __init__(
        self,
        model_name: str = MODEL_NAME,
        max_new_tokens: int = MAX_NEW_TOKENS,
    ) -> None:


        self.pipe = load_qwen()
        self.generation_config = og.GenerationConfig()
        self.generation_config.max_new_tokens = max_new_tokens
        # Prompts reach the pipeline already templated.
        self.generation_config.apply_chat_template = False

    def _build_conversation(
        self, result: MinimalSearchResults
    ) -> List[Dict[str, str]]:
        context = "\n\n".join(
            f"[{source.file_path}]\n{source.content}"
            for source in result.retrieved_sources[:SOURCES_INCLUDED]
        )
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Context:\n{context}\n\nQuestion: {result.question}"
                ),
            },
        ]

    def answer(self, result: MinimalSearchResults) -> str:
        answers: List[str] = self.answer_batch([result])
        return answers[0]

    def answer_batch(self, results: List[MinimalSearchResults]) -> List[str]:
        """Generates one answer per search result with the OpenVINO GenAI
        pipeline.

        Args:
            results: Questions with their retrieved sources.

        Returns:
            The answers, in the same order as results.
        """
        if not results:
            return []

        self.tokenizer = self.pipe.get_tokenizer()

        def _count(self, text: str) -> int:
            return int(self.tokenizer.encode(text).input_ids.shape[1])

        def _truncate(self, text: str, max_tokens: int) -> str:
            ids = self.tokenizer.encode(text).input_ids.data[0]
            if len(ids) <= max_tokens:
                return text
            return str(self.tokenizer.decode(ids[:max_tokens].tolist()))

        prompts: List[str] = [
            self.tokenizer.apply_chat_template(
                history=self._build_conversation(result),
                add_generation_prompt=True,
            )
            for result in results
        ]
        # A list of prompts is scheduled by the pipeline itself
        # (continuous batching): no padding, and each prompt stops at its
        # own end-of-sequence. ~25% faster than one call per prompt.
        decoded = self.pipe.generate(prompts, self.generation_config)
        return [text.strip() for text in decoded.texts]
