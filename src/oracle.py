"""Answer generation with Qwen3-0.6B, run by OpenVINO GenAI."""
import os
import shutil
from pathlib import Path
from typing import Dict, List

import openvino_genai as og
from src.models import MinimalSearchResults

MODEL_NAME = "Qwen/Qwen3-0.6B"
QWEN_OV_DIR = Path("data/models/qwen3-0.6b-ov-int8")
# Compiled models are cached here by OpenVINO (shared with the embedder):
# loading Qwen goes from ~30 s to ~5 s once the cache exists.
OV_CACHE_DIR = Path("data/models/ov_cache")
MAX_NEW_TOKENS = 128
SYSTEM_PROMPT = (
    "Answer the question using the provided context. If the "
    "context does not relate to the answer, say so explicitly. Be "
    "concise and factual, answer in 2-3 phrases."
)
SOURCES_INCLUDED = 3
# Sources are at most 2000 characters (~600 tokens) each, so the context
# is bounded already; the question is not, and a huge one would keep the
# CPU busy for minutes.
MAX_QUESTION_TOKENS = 256
# Files only present once an export is complete.
_EXPORT_MARKERS = (
    "openvino_model.xml",
    "openvino_tokenizer.xml",
    "openvino_detokenizer.xml",
    "tokenizer_config.json",
)


def export_qwen() -> None:
    """Exports Qwen3-0.6B to OpenVINO with int8 weights, unless done.

    The export is written to a temporary directory renamed into place
    once complete, so an interrupted first run is redone next time.
    """
    if all((QWEN_OV_DIR / name).exists() for name in _EXPORT_MARKERS):
        return
    from optimum.intel import OVModelForCausalLM, OVWeightQuantizationConfig
    from transformers import AutoTokenizer
    import openvino as ov
    from openvino_tokenizers import convert_tokenizer

    print("Exporting Qwen3-0.6B to OpenVINO int8 (first run only)...")
    tmp_dir = QWEN_OV_DIR.with_name(QWEN_OV_DIR.name + ".tmp")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    model = OVModelForCausalLM.from_pretrained(
        MODEL_NAME,
        export=True,
        quantization_config=OVWeightQuantizationConfig(bits=8),
    )
    model.save_pretrained(tmp_dir)
    # LLMPipeline needs the tokenizer in OpenVINO format too, which
    # OVModelForCausalLM.save_pretrained() does not write.
    hf_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    ov_tokenizer, ov_detokenizer = convert_tokenizer(
        hf_tokenizer, with_detokenizer=True
    )
    ov.save_model(ov_tokenizer, tmp_dir / "openvino_tokenizer.xml")
    ov.save_model(ov_detokenizer, tmp_dir / "openvino_detokenizer.xml")
    hf_tokenizer.save_pretrained(tmp_dir)
    shutil.rmtree(QWEN_OV_DIR, ignore_errors=True)
    os.replace(tmp_dir, QWEN_OV_DIR)


class Oracle:
    """Generates answers grounded in retrieved sources."""

    def __init__(self, max_new_tokens: int = MAX_NEW_TOKENS) -> None:
        """Loads the model, exporting it on first use.

        Args:
            max_new_tokens: Maximum length of an answer, in tokens.
        """
        export_qwen()
        OV_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.pipe = og.LLMPipeline(
            str(QWEN_OV_DIR), "CPU", CACHE_DIR=str(OV_CACHE_DIR)
        )
        self.tokenizer = self.pipe.get_tokenizer()
        self.generation_config = og.GenerationConfig()
        self.generation_config.max_new_tokens = max_new_tokens
        # Prompts reach the pipeline already templated.
        self.generation_config.apply_chat_template = False

    def _truncate(self, text: str, max_tokens: int) -> str:
        """Cuts text to at most max_tokens tokens."""
        ids = self.tokenizer.encode(text).input_ids.data[0]
        if len(ids) <= max_tokens:
            return text
        return str(self.tokenizer.decode(ids[:max_tokens].tolist()))

    def _build_conversation(
        self, result: MinimalSearchResults
    ) -> List[Dict[str, str]]:
        """Builds the chat messages: best sources, then the question."""
        context = "\n\n".join(
            f"[{source.file_path}]\n{source.content}"
            for source in result.retrieved_sources[:SOURCES_INCLUDED]
        )
        question = self._truncate(result.question, MAX_QUESTION_TOKENS)
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}",
            },
        ]

    def answer(self, result: MinimalSearchResults) -> str:
        """Generates the answer for a single search result."""
        return self.answer_batch([result])[0]

    def answer_batch(self, results: List[MinimalSearchResults]) -> List[str]:
        """Generates one answer per search result.

        Args:
            results: Questions with their retrieved sources.

        Returns:
            The answers, in the same order as results.
        """
        if not results:
            return []
        prompts: List[str] = [
            self.tokenizer.apply_chat_template(
                self._build_conversation(result),
                add_generation_prompt=True,
                # Qwen3 would otherwise spend its token budget "thinking".
                extra_context={"enable_thinking": False},
            )
            for result in results
        ]
        # A list of prompts is scheduled by the pipeline itself
        # (continuous batching): no padding, and each prompt stops at its
        # own end-of-sequence. ~25% faster than one call per prompt.
        decoded = self.pipe.generate(prompts, self.generation_config)
        return [text.strip() for text in decoded.texts]
