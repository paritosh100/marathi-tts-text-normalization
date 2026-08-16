"""Batch eval: normalize -> synthesize -> transcribe -> WER against reference output."""

import argparse
import json
import logging
from pathlib import Path

from normalize import normalize_text
from transcribe import transcribe
from tts_client import SpeechSynthesisError, synthesize_speech

logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).parent
EVAL_PATH = SCRIPT_DIR / "dataset" / "eval.jsonl"
REPORT_PATH = SCRIPT_DIR / "eval_report.jsonl"


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
            try:
                normalized = normalize_text(raw)
                audio = synthesize_speech(normalized)
                transcribed = transcribe(audio)
                wer = word_error_rate(reference, transcribed)
                record = {
                    "input": raw,
                    "reference": reference,
                    "normalized": normalized,
                    "transcribed": transcribed,
                    "wer": wer,
                }
            except SpeechSynthesisError as e:
                record = {"input": raw, "reference": reference, "error": str(e)}
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            results.append(record)
            logger.info("[%d/%d] wer=%s", idx, len(rows), record.get("wer", "error"))

    scored = [r["wer"] for r in results if "wer" in r]
    aggregate = sum(scored) / len(scored) if scored else float("nan")
    print(f"Evaluated {len(results)} rows ({len(scored)} scored, {len(results) - len(scored)} errored)")
    print(f"Aggregate WER: {aggregate:.4f}")
    return aggregate


def _demo():
    assert word_error_rate("a b c", "a b c") == 0.0
    assert word_error_rate("a b c", "a b") == 1 / 3
    assert word_error_rate("a b c", "") == 1.0
    assert word_error_rate("", "") == 0.0
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
