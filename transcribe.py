"""Marathi ASR transcription via mlx-whisper, for round-trip WER evaluation."""

import logging
import tempfile
from pathlib import Path

import mlx_whisper

logger = logging.getLogger(__name__)

WHISPER_MODEL = "mlx-community/whisper-large-v3-mlx"


def transcribe(audio_bytes: bytes) -> str:
    """Transcribe synthesized audio bytes back to text via Whisper."""
    with tempfile.NamedTemporaryFile(suffix=".mp3") as f:
        f.write(audio_bytes)
        f.flush()
        result = mlx_whisper.transcribe(
            f.name, path_or_hf_repo=WHISPER_MODEL, language="mr"
        )
    return result["text"].strip()


def _demo():
    from tts_client import synthesize_speech

    audio = synthesize_speech("नमस्कार, आज हवामान छान आहे.")
    text = transcribe(audio)
    assert text, "transcribe returned empty text"
    print("Transcribed:", text)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _demo()
