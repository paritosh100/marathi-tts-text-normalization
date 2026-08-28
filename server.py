"""Local-only FastAPI server backing the site/index.html Speak demo.

Binds to 127.0.0.1 only: single-user local tool, never exposed to the
network, so no auth layer is implemented (see specs/sarvam_pipeline_design.md).
"""

import base64
import logging
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from normalize import normalize_text
from tts_client import SpeechSynthesisError, synthesize_speech

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SITE_DIR = Path(__file__).parent / "site"
MAX_TEXT_LENGTH = 500

app = FastAPI()
app.mount("/assets", StaticFiles(directory=SITE_DIR / "assets"), name="assets")


class SpeakRequest(BaseModel):
    text: str
    engine: str = "edge"  # or "indicf5"


class SpeakResponse(BaseModel):
    normalized_text: str
    audio_url: str


@app.get("/")
def index():
    return FileResponse(SITE_DIR / "index.html")


@app.post("/api/speak", response_model=SpeakResponse)
def speak(req: SpeakRequest):
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text must not be empty.")
    if len(text) > MAX_TEXT_LENGTH:
        raise HTTPException(status_code=400, detail=f"Text must be at most {MAX_TEXT_LENGTH} characters.")

    start = time.monotonic()
    try:
        normalized = normalize_text(text)
    except Exception:
        logger.exception("normalize_failed latency=%.2fs", time.monotonic() - start)
        raise HTTPException(status_code=500, detail="Couldn't normalize the text — try again.")

    try:
        if req.engine == "indicf5":
            from indicf5_client import synthesize_speech_indicf5

            audio_bytes = synthesize_speech_indicf5(normalized)  # wav, not mp3
            mime_type = "audio/wav"
        else:
            audio_bytes = synthesize_speech(normalized)
            mime_type = "audio/mpeg"
    except SpeechSynthesisError:
        logger.error("synthesize_failed latency=%.2fs", time.monotonic() - start)
        raise HTTPException(status_code=502, detail="Couldn't generate speech — try again.")

    logger.info("speak_ok latency=%.2fs", time.monotonic() - start)
    audio_b64 = base64.b64encode(audio_bytes).decode("ascii")
    return SpeakResponse(
        normalized_text=normalized,
        audio_url=f"data:{mime_type};base64,{audio_b64}",
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
