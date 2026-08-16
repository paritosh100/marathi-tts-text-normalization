"""Marathi speech normalization via the fine-tuned LoRA checkpoint (mlx-lm)."""

import logging
import os
import re

from mlx_lm import generate, load
from mlx_lm.sample_utils import make_logits_processors

logger = logging.getLogger(__name__)

BASE_MODEL = "unsloth/Meta-Llama-3.1-8B"
ADAPTER_PATH = os.environ.get(
    "NORMALIZE_ADAPTER_PATH",
    os.path.expanduser(
        "~/.unsloth/studio/exports/unsloth_Meta-Llama-3.1-8B_1785626973/checkpoint-450"
    ),
)
REPETITION_PENALTY = 1.3
MAX_TOKENS = 512

# Must match the Alpaca-format training data exactly (dataset/generate_dataset.py) —
# the LoRA was never trained on chat-template tokens, so apply_chat_template() prompts
# it into a format it never learned an EOS for, causing runaway/repeated generation.
INSTRUCTION = (
    "Convert the raw Marathi text into phonetically normalized, speech-ready "
    "Devanagari text with pause annotations for TTS."
)
PROMPT_TEMPLATE = """### Instruction:
{instruction}

### Input:
{input}

### Response:
"""

# Defensive net: checkpoint-450 doesn't always emit EOS cleanly, so truncate at the
# first sign of the model hallucinating another turn instead of trusting max_tokens alone.
STOP_MARKERS = ("### Instruction:", "### Input:", "### Response:")

# checkpoint-450 also sometimes keeps going *without* leaking a template marker —
# repeating the sentence or inventing an unrelated follow-on one. Since every raw
# input here is prompt-engineered as a single sentence, its own terminal punctuation
# count (., !, ? — not decimal points, which are never followed by whitespace) is the
# expected count in the response too; anything past that is runaway generation.
# ponytail: sentence-count heuristic, not real stopping-criteria/EOS control. Assumes
# raw is well-formed single/few-sentence text without embedded quotes or abbreviated
# periods. Upgrade path: more training steps so the model emits EOS reliably, or a
# proper stop-string/logits-based stopping criterion in generate() itself.
SENTENCE_END_RE = re.compile(r"[.!?](?=\s|$)")


def _truncate_runaway(raw: str, generated: str) -> str:
    expected = len(SENTENCE_END_RE.findall(raw))
    if expected == 0:
        return generated
    matches = list(SENTENCE_END_RE.finditer(generated))
    if len(matches) <= expected:
        return generated
    return generated[: matches[expected - 1].end()]

_model = None
_tokenizer = None


def _get_model():
    global _model, _tokenizer
    if _model is None:
        logger.info("Loading normalization model (base=%s, adapter=%s)", BASE_MODEL, ADAPTER_PATH)
        _model, _tokenizer = load(BASE_MODEL, adapter_path=ADAPTER_PATH)
    return _model, _tokenizer


def normalize_text(raw: str) -> str:
    """Convert raw Marathi text into normalized, pause-annotated Devanagari text."""
    model, tokenizer = _get_model()
    prompt = PROMPT_TEMPLATE.format(instruction=INSTRUCTION, input=raw)
    logits_processors = make_logits_processors(repetition_penalty=REPETITION_PENALTY)
    text = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=MAX_TOKENS,
        logits_processors=logits_processors,
    )
    for marker in STOP_MARKERS:
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx]
    text = _truncate_runaway(raw, text)
    return text.strip()


def _demo():
    out = normalize_text("मी काल २:३० PM ला ५०० रुपयांचे पुस्तक विकत घेतले.")
    assert out, "normalize_text returned empty output"
    print("Normalized:", out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _demo()
