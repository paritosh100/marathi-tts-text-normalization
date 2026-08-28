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

# checkpoint-450's decimal handling is unreliable: eval_report.jsonl showed it
# both misreading the integer part ("18.5" -> "अठरावीस पूर्णांक पाच", not अठरा)
# and switching decimal-separator words at random (पॉईंट/दशांश/पूर्णांक) — because
# the Gemini-generated training data itself uses all three inconsistently for
# the same "X.5" pattern. Numbers are a solved problem, so decimals are
# expanded to words deterministically before the raw text ever reaches the
# model, removing this from what the LLM has to get right. Currency amounts
# (₹-prefixed) are left alone since those follow a rupees/paise or lakh/crore
# convention the model already handles correctly, not a plain decimal reading.
_ONES = (
    "शून्य एक दोन तीन चार पाच सहा सात आठ नऊ "
    "दहा अकरा बारा तेरा चौदा पंधरा सोळा सतरा अठरा एकोणीस "
    "वीस एकवीस बावीस तेवीस चोवीस पंचवीस सव्वीस सत्तावीस अठ्ठावीस एकोणतीस "
    "तीस एकतीस बत्तीस तेहतीस चौतीस पस्तीस छत्तीस सदतीस अडतीस एकोणचाळीस "
    "चाळीस एक्केचाळीस बेचाळीस त्रेचाळीस चव्वेचाळीस पंचेचाळीस सेहेचाळीस सत्तेचाळीस अठ्ठेचाळीस एकोणपन्नास "
    "पन्नास एक्कावन्न बावन्न त्रेपन्न चोपन्न पंचावन्न छप्पन्न सत्तावन्न अठ्ठावन्न एकोणसाठ "
    "साठ एकसष्ट बासष्ट त्रेसष्ट चौसष्ट पासष्ट सहासष्ट सदुसष्ट अडुसष्ट एकोणसत्तर "
    "सत्तर एक्काहत्तर बाहत्तर त्र्याहत्तर चौऱ्याहत्तर पंच्याहत्तर शहात्तर सत्याहत्तर अठ्ठ्याहत्तर एकोणऐंशी "
    "ऐंशी एक्क्याऐंशी ब्याऐंशी त्र्याऐंशी चौऱ्याऐंशी पंच्याऐंशी शहाऐंशी सत्त्याऐंशी अठ्ठ्याऐंशी एकोणनव्वद "
    "नव्वद एक्क्याण्णव ब्याण्णव त्र्याण्णव चौऱ्याण्णव पंच्याण्णव शहाण्णव सत्त्याण्णव अठ्ठ्याण्णव नव्व्याण्णव"
).split()


def _int_to_marathi(n: int) -> str:
    """0-99999 -> Marathi words (हजार/शे composition), e.g. 105 -> 'एकशे पाच'."""
    if n < 100:
        return _ONES[n]
    parts = []
    if n >= 1000:
        parts.append(_ONES[n // 1000] + " हजार")
        n %= 1000
    if n >= 100:
        hundreds, n = n // 100, n % 100
        parts.append("शंभर" if hundreds == 1 and n == 0 else _ONES[hundreds] + "शे")
    if n > 0:
        parts.append(_ONES[n])
    return " ".join(parts)


_DEVANAGARI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")

# checkpoint-450 also drifts on the *magnitude* of large comma-grouped
# integers (eval_report.jsonl: "₹62,000" -> "ब्याऐंशी हजार" i.e. 82,000; a
# Devanagari-digit "₹६२,०००" came back as outright gibberish). Same fix as
# decimals: numbers are a solved problem, so expand them deterministically
# before the model sees them instead of trusting it to read digit grouping.
# Small numbers, times, and percentages aren't touched here because eval
# showed the model already reads those correctly -- this targets only the
# failure actually observed.
# ponytail: assumes Western 3-digit comma grouping (62,000). Indian lakh/crore
# grouping (1,00,000+) isn't handled -- not seen in eval data yet.
_COMMA_INT_RE = re.compile(r"(?<![.\d])[0-9०-९]{1,3}(?:,[0-9०-९]{3})+(?!\.[0-9])")


def _expand_comma_integers(text: str) -> str:
    def replace(match: re.Match) -> str:
        digits = match.group(0).translate(_DEVANAGARI_DIGITS).replace(",", "")
        return _int_to_marathi(int(digits))

    return _COMMA_INT_RE.sub(replace, text)


# ₹/$ can have a space before the amount ("₹ १०.५०"), which the currency
# exclusion below doesn't see -- its lookbehind only checks the character
# immediately before the digits. Eval showed that space letting a paise
# amount slip through and get wrongly decimal-expanded ("दहा पूर्णांक पाच शून्य"
# instead of being left for the model's rupees/paise reading). Collapse the
# space first so the exclusion always sees the currency symbol directly.
_CURRENCY_SPACE_RE = re.compile(r"([₹$])\s+(?=\d)")

_DECIMAL_RE = re.compile(r"(?<![₹$])\b(\d+)\.(\d+)\b")

# Marathi doesn't say "point five" for X.5 — it has dedicated half-idioms
# (दीड=1.5, अडीच=2.5, साडे+number=X.5 for X>=3), confirmed throughout
# dataset/train.jsonl (साडेतीन, साडेचार, साडेसात, साडेआठ, ...). Falling back to
# generic digit-by-digit "पूर्णांक" reading for these would be correct-but-
# unnatural and would actively mismatch the established convention.
# ponytail: साडे- covers whole<100 only (untested for hundreds+half); anything
# larger falls back to the generic reading below.
_HALF_WORDS = {0: "अर्धा", 1: "दीड", 2: "अडीच"}


def _expand_decimals(text: str) -> str:
    text = _CURRENCY_SPACE_RE.sub(r"\1", text)

    def replace(match: re.Match) -> str:
        whole, frac = match.groups()
        whole_n = int(whole)
        # frac may be in Devanagari digits ("५"); translate before comparing
        # so the साडे/दीड/अडीच idiom applies regardless of digit script --
        # eval showed "४२.५" silently skipping it while "42.5" didn't, purely
        # because this was comparing against the ASCII literal "5".
        if frac.translate(_DEVANAGARI_DIGITS) == "5":
            if whole_n in _HALF_WORDS:
                return _HALF_WORDS[whole_n]
            if whole_n < 100:
                return "साडे" + _ONES[whole_n]
        frac_words = " ".join(_ONES[int(d)] for d in frac)
        return f"{_int_to_marathi(whole_n)} पूर्णांक {frac_words}"

    return _DECIMAL_RE.sub(replace, text)

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

# ...but the AC/temperature eval case showed the model restarting the whole
# response after a "[pause]" tag instead of a period, which the punctuation
# count above never sees. Catch that verbatim-restart case separately: if the
# response's own opening words reappear later in the string, everything from
# there on is the repeat.
# ponytail: exact-match prefix repeat only (no fuzzy/near-duplicate detection),
# 3-word prefix chosen to avoid false-triggering on short common phrases.
_REPEAT_PREFIX_WORDS = 3
_TRAILING_SEPARATOR_RE = re.compile(r"[\s.,!?]*(\[pause\][\s.,!?]*)*$")


def _truncate_runaway(raw: str, generated: str) -> str:
    expected = len(SENTENCE_END_RE.findall(raw))
    if expected:
        matches = list(SENTENCE_END_RE.finditer(generated))
        if len(matches) > expected:
            generated = generated[: matches[expected - 1].end()]

    words = generated.split()
    if len(words) >= _REPEAT_PREFIX_WORDS * 2:
        prefix = " ".join(words[:_REPEAT_PREFIX_WORDS])
        restart = generated.find(prefix, len(prefix))
        if restart != -1:
            return _TRAILING_SEPARATOR_RE.sub("", generated[:restart])
    return generated

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
    preprocessed = _expand_comma_integers(_expand_decimals(raw))
    prompt = PROMPT_TEMPLATE.format(instruction=INSTRUCTION, input=preprocessed)
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
    assert _int_to_marathi(18) == "अठरा"
    assert _int_to_marathi(42) == "बेचाळीस"
    assert _int_to_marathi(105) == "एकशे पाच"
    assert _int_to_marathi(100) == "शंभर"
    assert _expand_decimals("साधारण 18.5 km/l इतका आहे") == "साधारण साडेअठरा km/l इतका आहे"
    assert _expand_decimals("₹105.50 प्रति लीटर") == "₹105.50 प्रति लीटर", "currency decimals must be left alone"
    assert _expand_decimals("1.5 kg बटाटे") == "दीड kg बटाटे"
    assert _expand_decimals("2.5% वाढ") == "अडीच% वाढ"
    assert _expand_decimals("92.4% score") == "ब्याण्णव पूर्णांक चार% score"
    assert _expand_decimals("४२.५°C पर्यंत") == "साडेबेचाळीस°C पर्यंत", "half-idiom must apply for Devanagari digits too"
    assert (
        _expand_decimals("भाव ₹ १०.५० ने वाढले") == "भाव ₹१०.५० ने वाढले"
    ), "currency decimals must be left alone even with a space before the amount"
    assert _expand_comma_integers("₹62,000 आहे") == "₹बासष्ट हजार आहे"
    assert _expand_comma_integers("₹६२,००० आहे") == "₹बासष्ट हजार आहे"
    assert _expand_comma_integers("₹45,999 आहे") == "₹पंचेचाळीस हजार नऊशे नव्व्याण्णव आहे"
    assert _expand_comma_integers("₹105.50 प्रति लीटर") == "₹105.50 प्रति लीटर", "plain decimals must be left alone"
    assert (
        _truncate_runaway(
            "कृपया AC चे Temperature २२ degree वर सेट कर.",
            "कृपया ए सी चे टेम्परेचर बावीस अंश डिग्रीज वर सेट कर [pause] कृपया ए सी चे टेम्परेचर बावीस अंश डिग्रीज वर सेट कर.",
        )
        == "कृपया ए सी चे टेम्परेचर बावीस अंश डिग्रीज वर सेट कर"
    )
    print("normalize_text helpers self-check passed")

    out = normalize_text("मी काल २:३० PM ला ५०० रुपयांचे पुस्तक विकत घेतले.")
    assert out, "normalize_text returned empty output"
    print("Normalized:", out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _demo()
