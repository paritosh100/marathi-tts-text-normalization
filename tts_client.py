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

# edge_tts.Communicate always XML-escapes its input before embedding it in SSML,
# so a literal <break> tag isn't an option — it would just get read out loud.
# A comma is the only [pause] -> real-pause translation neural voices honor here.
_PAUSE_RE = re.compile(r"\s*\[pause\]\s*")
_DOUBLE_PUNCT_RE = re.compile(r"([,.!?])\s*,")


def _apply_pauses(text: str) -> str:
    text = _PAUSE_RE.sub(", ", text)
    text = _DOUBLE_PUNCT_RE.sub(r"\1", text)
    return text.strip()


class SpeechSynthesisError(Exception):
    pass


async def _synthesize_once(text: str, voice: str) -> bytes:
    communicate = edge_tts.Communicate(_apply_pauses(text), voice)
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
    assert _apply_pauses("अरे यार, [pause] काय झालं?") == "अरे यार, काय झालं?"
    assert _apply_pauses("हा प्रोजेक्ट [pause] पूर्ण होईल.") == "हा प्रोजेक्ट, पूर्ण होईल."
    audio = synthesize_speech("नमस्कार, [pause] आज हवामान छान आहे.")
    assert isinstance(audio, bytes) and len(audio) > 0
    print(f"Got {len(audio)} bytes of audio")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _demo()
