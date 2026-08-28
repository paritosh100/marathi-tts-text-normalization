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


def audio_filename(text: str, ext: str = "mp3") -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16] + f".{ext}"


def speak(
    raw_text: str, voice: str = DEFAULT_VOICE, output_dir: Path = OUTPUT_DIR, engine: str = "edge"
) -> tuple[str, Path]:
    normalized = normalize_text(raw_text)
    if engine == "indicf5":
        from indicf5_client import synthesize_speech_indicf5

        audio = synthesize_speech_indicf5(normalized)  # wav bytes, not mp3
        ext = "wav"
    else:
        audio = synthesize_speech(normalized, voice)
        ext = "mp3"
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / audio_filename(normalized, ext)
    path.write_bytes(audio)
    return normalized, path


def main():
    parser = argparse.ArgumentParser(description="Normalize Marathi text and synthesize speech.")
    parser.add_argument("text", help="Raw Marathi text to speak")
    parser.add_argument("--voice", default=DEFAULT_VOICE, help="edge-tts voice (ignored for --engine indicf5)")
    parser.add_argument("--engine", choices=["edge", "indicf5"], default="edge")
    args = parser.parse_args()

    try:
        normalized, path = speak(args.text, args.voice, engine=args.engine)
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
