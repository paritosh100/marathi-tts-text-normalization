"""edge-tts wrapper with bounded retry for transient failures."""

import asyncio
import logging
import re
import time

import edge_tts

logger = logging.getLogger(__name__)

DEFAULT_VOICE = "mr-IN-AarohiNeural"
MAX_RETRIES = 3
BACKOFF_SECONDS = 1.5

# Shared by every synthesis engine (edge-tts, indicf5_client): none of them
# understand the literal "[pause]" marker normalize_text() emits -- edge-tts
# would read the bracket text aloud, and IndicF5 chokes on it entirely
# (degenerates into repeating a garbage token). A comma is the one
# [pause] -> real-pause translation every plain-text TTS engine we've used
# honors.
_PAUSE_RE = re.compile(r"\s*\[pause\]\s*")
_DOUBLE_PUNCT_RE = re.compile(r"([,.!?])\s*,")


def apply_pauses(text: str) -> str:
    text = _PAUSE_RE.sub(", ", text)
    text = _DOUBLE_PUNCT_RE.sub(r"\1", text)
    return text.strip()


class SpeechSynthesisError(Exception):
    pass


async def _synthesize_once(text: str, voice: str) -> bytes:
    communicate = edge_tts.Communicate(apply_pauses(text), voice)
    chunks = bytearray()
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            chunks.extend(chunk["data"])
    return bytes(chunks)


def synthesize_speech(text: str, voice: str = DEFAULT_VOICE) -> bytes:
    """Synthesize speech audio for text, retrying transient edge-tts failures."""
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        start = time.monotonic()
        try:
            audio = asyncio.run(_synthesize_once(text, voice))
            latency = time.monotonic() - start
            if not audio:
                raise SpeechSynthesisError("edge-tts returned empty audio")
            logger.info("tts_call status=ok attempt=%d latency=%.2fs", attempt, latency)
            return audio
        except Exception as e:
            last_error = e
            latency = time.monotonic() - start
            logger.warning(
                "tts_call status=error attempt=%d latency=%.2fs error=%s",
                attempt, latency, type(e).__name__,
            )
            if attempt < MAX_RETRIES:
                time.sleep(BACKOFF_SECONDS * attempt)
    raise SpeechSynthesisError(
        f"edge-tts failed after {MAX_RETRIES} attempts: {last_error}"
    ) from last_error


def _demo():
    assert apply_pauses("अरे यार, [pause] काय झालं?") == "अरे यार, काय झालं?"
    assert apply_pauses("हा प्रोजेक्ट [pause] पूर्ण होईल.") == "हा प्रोजेक्ट, पूर्ण होईल."
    audio = synthesize_speech("नमस्कार, [pause] आज हवामान छान आहे.")
    assert isinstance(audio, bytes) and len(audio) > 0
    print(f"Got {len(audio)} bytes of audio")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _demo()
