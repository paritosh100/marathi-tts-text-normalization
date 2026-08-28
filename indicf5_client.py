"""IndicF5 (AI4Bharat) zero-shot Marathi voice synthesis.

Unlike tts_client.py's edge-tts (a named, off-the-shelf voice), IndicF5 clones
a voice from a short reference clip + its exact transcript -- see
REFERENCE_AUDIO_PATH/REFERENCE_TEXT below.

Uses a vendored copy of f5_tts (vendor/indicf5_f5_tts/), not the official
f5-tts PyPI package (currently 1.1.22). That package's `infer_process`
deterministically degenerates into repeating a garbage token for a large
fraction of inputs when paired with IndicF5's checkpoint (confirmed: same
text, same reference, byte-identical broken output every retry) -- a real
regression versus AI4Bharat's own bundled f5_tts snapshot (an older,
unversioned copy), which produces correct output for the exact same inputs.
Root cause not further diagnosed (a diff against upstream f5-tts's inference
changes wasn't worth the time here); the known-working snapshot is used
instead. See PROGRESS.md's Stage 2 Phase 3 entry for how this was found.

Getting even the vendored snapshot running exposed a string of real bugs/
environment gaps in AI4Bharat's own code, patched directly in the vendored
copy (not here) -- see PROGRESS.md's Stage 2 Phase 0 entry for that trail.
"""

import io
import logging
import sys
from pathlib import Path

import torch

from tts_client import SpeechSynthesisError, apply_pauses

logger = logging.getLogger(__name__)

VENDOR_DIR = Path(__file__).parent / "vendor" / "indicf5_f5_tts"
sys.path.insert(0, str(VENDOR_DIR))

from f5_tts.infer.utils_infer import (  # noqa: E402 (must follow sys.path.insert)
    infer_process,
    load_model,
    load_vocoder,
    preprocess_ref_audio_text,
)
from f5_tts.model import DiT  # noqa: E402

MODEL_REPO = "ai4bharat/IndicF5"
# Standard F5-TTS "Base" architecture (matches f5_tts's own F5TTS_Base_train.yaml) --
# confirmed correct empirically: the checkpoint loads with these dims and no
# shape mismatches.
_MODEL_CONFIG = dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4)

ASSETS_DIR = Path(__file__).parent / "assets"
REFERENCE_AUDIO_PATH = ASSETS_DIR / "reference_voice.wav"
# ai4bharat/indicvoices_r (CC-BY-4.0), Marathi/test-00000-of-00002.parquet row 32:
# a calm "Read"-scenario assistant-command sentence, cer=0.0 (dataset-verified
# transcript, not ASR-guessed), high SNR. IndicF5's own shipped Marathi prompt
# clip was tagged "happy" and made every synthesis sound like shouting --
# swapped for this one. See PROGRESS.md for how it was picked.
REFERENCE_TEXT = "मी सिट्रस वापरून श्रीराम फायनान्स कंपनीमधील माझे शैक्षणिक कर्ज परत करू शकतो का"

_DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

_model = None
_vocoder = None
_ref_audio = None
_ref_text = None


def _get_model():
    global _model, _vocoder, _ref_audio, _ref_text
    if _model is None:
        from huggingface_hub import hf_hub_download

        logger.info("Loading IndicF5 (device=%s)", _DEVICE)
        ckpt_path = hf_hub_download(MODEL_REPO, filename="model.safetensors")
        vocab_path = hf_hub_download(MODEL_REPO, filename="checkpoints/vocab.txt")
        _vocoder = load_vocoder(vocoder_name="vocos", is_local=False, device=_DEVICE)
        _model = load_model(
            DiT,
            _MODEL_CONFIG,
            ckpt_path=ckpt_path,
            mel_spec_type="vocos",
            vocab_file=vocab_path,
            device=_DEVICE,
        )
        _ref_audio, _ref_text = preprocess_ref_audio_text(str(REFERENCE_AUDIO_PATH), REFERENCE_TEXT)
    return _model, _vocoder, _ref_audio, _ref_text


def synthesize_speech_indicf5(text: str) -> bytes:
    """Synthesize speech audio for text via IndicF5 zero-shot voice cloning."""
    model, vocoder, ref_audio, ref_text = _get_model()
    try:
        audio, sample_rate, _ = infer_process(
            ref_audio, ref_text, apply_pauses(text), model, vocoder, mel_spec_type="vocos", device=_DEVICE,
        )
    except Exception as e:
        raise SpeechSynthesisError(f"IndicF5 synthesis failed: {e}") from e
    if audio is None or len(audio) == 0:
        raise SpeechSynthesisError("IndicF5 returned empty audio")

    buffer = io.BytesIO()
    import soundfile as sf

    sf.write(buffer, audio, samplerate=sample_rate, format="WAV")
    return buffer.getvalue()


def _demo():
    # "माझ्याकडे पन्नास रुपये आहेत." specifically: eval found the official
    # f5-tts package degenerating into a repeated garbage token on this exact
    # short, plain sentence -- no [pause], no special formatting. Confirms
    # the vendored copy doesn't have that regression.
    audio = synthesize_speech_indicf5("माझ्याकडे पन्नास रुपये आहेत.")
    assert isinstance(audio, bytes) and len(audio) > 0
    print(f"Got {len(audio)} bytes of audio")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _demo()
