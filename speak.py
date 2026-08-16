"""CLI: raw Marathi text -> normalized text -> synthesized audio file."""

import argparse
import hashlib
import logging
import sys
import time
from pathlib import Path

from normalize import normalize_text
from tts_client import DEFAULT_VOICE, SpeechSynthesisError, synthesize_speech

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent / "output"


def audio_filename(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16] + ".mp3"


def speak(raw_text: str, voice: str = DEFAULT_VOICE, output_dir: Path = OUTPUT_DIR) -> tuple[str, Path]:
    normalized = normalize_text(raw_text)
    audio = synthesize_speech(normalized, voice)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / audio_filename(normalized)
    path.write_bytes(audio)
    return normalized, path


def main():
    parser = argparse.ArgumentParser(description="Normalize Marathi text and synthesize speech.")
    parser.add_argument("text", help="Raw Marathi text to speak")
    parser.add_argument("--voice", default=DEFAULT_VOICE)
    args = parser.parse_args()

    try:
        normalized, path = speak(args.text, args.voice)
    except SpeechSynthesisError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Normalized: {normalized}")
    print(f"Audio saved: {path}")


def _demo():
    normalized, path = speak("मी काल संध्याकाळी ५ वाजता ऑफिसला गेले.")
    assert normalized and path.exists() and path.stat().st_size > 0
    print("speak() self-check passed:", path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) > 1 and sys.argv[1] == "--selfcheck":
        _demo()
    else:
        main()
