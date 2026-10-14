import os
import shutil
from pathlib import Path
from typing import Dict, List
from transformers import AutoTokenizer
from optimum.intel import OVModelForCausalLM, OVWeightQuantizationConfig
import openvino as ov
import openvino_genai as og
from openvino_tokenizers import convert_tokenizer
from src.models import MinimalSearchResults
from src.utils import NO_SOURCES_ANSWER


MODEL_NAME = "Qwen/Qwen3-0.6B"
QWEN_OV_DIR = Path("data/models/qwen3-0.6b-ov-int8")
MAX_NEW_TOKENS = 128
# Token budget of the prompt. 5 sources of up to 2000 characters take
# ~2500 tokens, so a normal prompt is never cut; the caps only protect
# the CPU from pathological inputs (a 22k-token question used to keep
# the model busy for more than 10 minutes).
CONTEXT_SOURCES = 5
MAX_CONTEXT_TOKENS = 3072
MAX_QUESTION_TOKENS = 256
SYSTEM_PROMPT = (
    "Answer the question using the provided context. If the "
    "context does not relate to the answer, say so explicitly. Be "
    "concise and factual, answer in 2-3 phrases."
)
# Files only written at the very end of a complete export.
_EXPORT_MARKERS = (
    "openvino_model.xml",
    "openvino_tokenizer.xml",
    "openvino_detokenizer.xml",
    "tokenizer_config.json",
)


def export_qwen(model_name: str = MODEL_NAME) -> None:
    """Exports the model to OpenVINO with int8 weights under QWEN_OV_DIR,
    with its tokenizer in both Hugging Face and OpenVINO formats, unless
    a complete export is already there.

    Everything is written to a temporary directory that is only renamed
    into place once complete, so an interrupted first run is redone on
    the next one instead of leaving a half-written model behind.
    """
    if all((QWEN_OV_DIR / name).exists() for name in _EXPORT_MARKERS):
        return
    print("Exporting Qwen3-0.6B to OpenVINO int8 (first run only)...")
    tmp_dir = QWEN_OV_DIR.with_name(QWEN_OV_DIR.name + ".tmp")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    model = OVModelForCausalLM.from_pretrained(
        model_name,
        export=True,
        quantization_config=OVWeightQuantizationConfig(bits=8),
    )
    model.save_pretrained(tmp_dir)
    # LLMPipeline needs the tokenizer in OpenVINO format too, which
    # OVModelForCausalLM.save_pretrained() does not write.
    hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
    ov_tokenizer, ov_detokenizer = convert_tokenizer(
        hf_tokenizer, with_detokenizer=True
    )
    ov.save_model(ov_tokenizer, tmp_dir / "openvino_tokenizer.xml")
    ov.save_model(ov_detokenizer, tmp_dir / "openvino_detokenizer.xml")
    hf_tokenizer.save_pretrained(tmp_dir)
    # Drop what an older, incomplete export may have left behind.
    shutil.rmtree(QWEN_OV_DIR, ignore_errors=True)
    os.replace(tmp_dir, QWEN_OV_DIR)


class Oracle:
    """Generates answers grounded in retrieved sources with Qwen3-0.6B,
    run by an OpenVINO GenAI pipeline."""

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        max_new_tokens: int = MAX_NEW_TOKENS,
    ) -> None:
        export_qwen(model_name)
        # Both load from disk: no network access once exported (loading
        # the tokenizer by Hub name queries the Hub API on every load).
        # transformers 4.57.3-4.57.6 mistakes any tokenizer it saved for a
        # Mistral one with a broken regex and warns about it; Qwen's is
        # fine, so declare explicitly that no regex fix is wanted.
        self.tokenizer = AutoTokenizer.from_pretrained(
            QWEN_OV_DIR, fix_mistral_regex=False
        )
        self.pipe = og.LLMPipeline(str(QWEN_OV_DIR), "CPU")
        self.generation_config = og.GenerationConfig()
        self.generation_config.max_new_tokens = max_new_tokens
        # Prompts reach the pipeline already templated.
        self.generation_config.apply_chat_template = False

    def _truncate(self, text: str, max_tokens: int) -> str:
        """Cuts text to at most max_tokens tokens."""
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(ids) <= max_tokens:
            return text
        return str(self.tokenizer.decode(ids[:max_tokens]))

    def _build_conversation(
        self, result: MinimalSearchResults
    ) -> List[Dict[str, str]]:
        """Builds the chat messages: the best sources, as many as fit in
        MAX_CONTEXT_TOKENS (the last one truncated to fit), then the
        question, itself capped to MAX_QUESTION_TOKENS."""
        blocks: List[str] = []
        budget = MAX_CONTEXT_TOKENS
        for source in result.retrieved_sources[:CONTEXT_SOURCES]:
            block = f"[{source.file_path}]\n{source.content}"
            size = len(self.tokenizer.encode(block, add_special_tokens=False))
            if size > budget:
                blocks.append(self._truncate(block, budget))
                break
            blocks.append(block)
            budget -= size
        context = "\n\n".join(blocks)
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
        answers: List[str] = self.answer_batch([result])
        return answers[0]

    def answer_batch(self, results: List[MinimalSearchResults]) -> List[str]:
        """Generates one answer per search result with the OpenVINO GenAI
        pipeline. A result without any source gets NO_SOURCES_ANSWER
        instead of an answer the model would make up from nothing.

        Args:
            results: Questions with their retrieved sources.

        Returns:
            The answers, in the same order as results.
        """
        answers = [NO_SOURCES_ANSWER] * len(results)
        grounded = [i for i, r in enumerate(results) if r.retrieved_sources]
        if not grounded:
            return answers

        prompts: List[str] = [
            self.tokenizer.apply_chat_template(
                self._build_conversation(results[i]),
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            for i in grounded
        ]
        # A list of prompts is scheduled by the pipeline itself
        # (continuous batching): no padding, and each prompt stops at its
        # own end-of-sequence. ~25% faster than one call per prompt.
        decoded = self.pipe.generate(prompts, self.generation_config)
        for i, text in zip(grounded, decoded.texts):
            answers[i] = text.strip()
        return answers
