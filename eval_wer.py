"""Batch eval: normalize -> synthesize -> transcribe -> WER against reference output."""

import argparse
import json
import logging
import re
from pathlib import Path

from normalize import normalize_text
from transcribe import transcribe
from tts_client import SpeechSynthesisError, synthesize_speech

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent
EVAL_PATH = SCRIPT_DIR / "dataset" / "eval.jsonl"
REPORT_PATH = SCRIPT_DIR / "eval_report.jsonl"

# tts_client._apply_pauses turns "[pause]" into a comma, and Whisper regularly
# hallucinates the literal word "pause" into the resulting silence (confirmed
# across eval_report.jsonl: पाउज/पाज/पॉज/पौज, always standalone at pause points,
# never a real word). Neither side of the comparison should count that: the
# reference's "[pause]" markup was never spoken either, so strip both before
# scoring rather than letting an ASR silence artifact inflate WER.
# ponytail: fixed hallucination token list from observed data, extend if new
# spellings show up.
_PAUSE_TAG_RE = re.compile(r"\[pause\]")
_PAUSE_HALLUCINATION_TOKENS = {"पाउज", "पाज", "पॉज", "पौज"}


def _strip_pause_artifacts(reference: str, hypothesis: str) -> tuple[str, str]:
    reference = _PAUSE_TAG_RE.sub("", reference)
    hyp_words = [w for w in hypothesis.split() if w.strip(".,।!?") not in _PAUSE_HALLUCINATION_TOKENS]
    return reference, " ".join(hyp_words)


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Word-level edit distance WER: substitutions+insertions+deletions / len(reference)."""
    ref = reference.split()
    hyp = hypothesis.split()
    n, m = len(ref), len(hyp)
    if n == 0:
        return 0.0 if m == 0 else 1.0

    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev_diag = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            temp = dp[j]
            if ref[i - 1] == hyp[j - 1]:
                dp[j] = prev_diag
            else:
                dp[j] = 1 + min(prev_diag, dp[j - 1], dp[j])
            prev_diag = temp
    return dp[m] / n


def run_eval(eval_path: Path = EVAL_PATH, report_path: Path = REPORT_PATH, limit: int | None = None):
    rows = []
    with open(eval_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    if limit:
        rows = rows[:limit]

    results = []
    with open(report_path, "w", encoding="utf-8") as out:
        for idx, row in enumerate(rows, 1):
            raw = row["input"]
            reference = row["output"]
            normalized = normalize_text(raw)
            # Text-only, no TTS/Whisper involved: isolates the normalizer's own
            # accuracy from audio round-trip noise (TTS mispronunciation, ASR
            # mishearing) that also feeds into `wer` below.
            normalized_wer = word_error_rate(reference, normalized)
            try:
                audio = synthesize_speech(normalized)
                transcribed = transcribe(audio)
                scored_reference, scored_transcribed = _strip_pause_artifacts(reference, transcribed)
                wer = word_error_rate(scored_reference, scored_transcribed)
                record = {
                    "input": raw,
                    "reference": reference,
                    "normalized": normalized,
                    "normalized_wer": normalized_wer,
                    "transcribed": transcribed,
                    "wer": wer,
                }
            except SpeechSynthesisError as e:
                record = {
                    "input": raw,
                    "reference": reference,
                    "normalized": normalized,
                    "normalized_wer": normalized_wer,
                    "error": str(e),
                }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            results.append(record)
            logger.info(
                "[%d/%d] normalized_wer=%s wer=%s",
                idx, len(rows), record.get("normalized_wer"), record.get("wer", "error"),
            )

    normalized_scored = [r["normalized_wer"] for r in results if "normalized_wer" in r]
    scored = [r["wer"] for r in results if "wer" in r]
    aggregate = sum(scored) / len(scored) if scored else float("nan")
    normalized_aggregate = sum(normalized_scored) / len(normalized_scored) if normalized_scored else float("nan")
    print(f"Evaluated {len(results)} rows ({len(scored)} scored, {len(results) - len(scored)} errored)")
    print(f"Aggregate normalizer-only WER (text vs text, no TTS/ASR): {normalized_aggregate:.4f}")
    print(f"Aggregate round-trip WER (text -> speech -> text): {aggregate:.4f}")
    return aggregate


def _demo():
    assert word_error_rate("a b c", "a b c") == 0.0
    assert word_error_rate("a b c", "a b") == 1 / 3
    assert word_error_rate("a b c", "") == 1.0
    assert word_error_rate("", "") == 0.0

    ref, hyp = _strip_pause_artifacts("रमेश, [pause] तुझा प्रोजेक्ट", "रमेश, पॉज, तुझा प्रोजेक्ट")
    assert ref == "रमेश,  तुझा प्रोजेक्ट", ref
    assert hyp == "रमेश, तुझा प्रोजेक्ट", hyp
    assert word_error_rate(ref, hyp) == 0.0

    print("word_error_rate self-check passed")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--selfcheck", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Only evaluate the first N rows")
    args = parser.parse_args()
    if args.selfcheck:
        _demo()
    else:
        run_eval(limit=args.limit)
