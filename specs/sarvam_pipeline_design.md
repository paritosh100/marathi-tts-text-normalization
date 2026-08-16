# Feature: Marathi Normalization → TTS → Whisper WER Eval Pipeline

## Scope note

This project has two kinds of outstanding work: ML/data work (dataset cleanup, retraining, git hygiene) and one genuine "build a feature" piece — wiring the fine-tuned normalization model into a real TTS voice and measuring quality with numbers. This doc scopes the fullstack-guardian plan to that second piece. The dataset cleanup is a separate, non-fullstack task.

## Voice backend decision: `edge-tts`, not Sarvam

Switched from the original plan's Sarvam API to `edge-tts` (`pip install edge-tts`) — a Python package that calls Microsoft Edge's TTS service directly. No API key, no signup, no rate limit, genuinely free, and it has a real Marathi neural voice (`mr-IN-AarohiNeural`). This removes an entire class of concern from the plan: no credential to protect, no `.env` entry, no auth-related security checklist items. Sarvam remains a documented fallback (see below) since it does have a real free tier, but isn't the default.

## Requirements (EARS Format)

- While the fine-tuned LoRA checkpoint (`checkpoint-450`, base `unsloth/Meta-Llama-3.1-8B`) is available locally, when a raw Marathi sentence is submitted, the system shall produce normalized Devanagari text with `[pause]` annotations, using the exact Alpaca template and `repetition_penalty=1.3` already validated earlier in this project.
- While normalized text has been produced, when audio is requested, the system shall call `edge-tts` with the `mr-IN-AarohiNeural` voice and save the returned audio to disk.
- While a batch of reference test sentences exists (`eval.jsonl`, post-cleanup), when the eval command runs, the system shall synthesize audio for each, transcribe it back via Whisper ASR, and compute Word Error Rate (WER) against the reference normalized text, producing a per-row and aggregate report.
- While `edge-tts` returns an error or empty audio (e.g. transient network failure — it's an unofficial, unauthenticated endpoint, not a paid SLA), when a TTS call is attempted, the system shall fail with a clear, actionable error rather than silently producing empty/broken audio, and retry a bounded number of times before giving up.
- (Fallback path, not built by default) If `edge-tts` output quality or reliability turns out to be insufficient, `sarvam_client.py` can be added later using Sarvam's free credit tier — the credential-handling checklist below would then apply to that module specifically.



## Architecture



### [Frontend]

Required — a real UI to try the pipeline, not just a CLI. Single local page, single purpose, no build tooling:

- One text input (raw Marathi text) + a "Speak" button.
- Client-side validation: button disabled while input is empty/whitespace-only (never the only guard — server re-validates).
- Loading state while the request is in flight (normalize + synthesize can take a few seconds on the LoRA inference step).
- On success: shows the normalized text (so you can see what the model changed) and an inline `<audio controls>` player with the synthesized speech, autoplaying once ready.
- On failure: a clear inline error message (e.g. "Couldn't generate speech — try again"), never a raw stack trace or silent blank state.
- Lives at `site/index.html` (already exists as a static page) — turn its current static demo into this live one, backed by the local API below. Plain HTML/JS, no framework — this is a single page with one interaction, a framework would be pure overhead here.



### [Backend]

Scripts reusing existing patterns already in this repo, plus a thin local API server for the UI:

- `normalize.py` — wraps the already-validated `mlx_lm.load()` / `generate()` call: `base=unsloth/Meta-Llama-3.1-8B`, `adapter_path=<checkpoint-450 export>`, the Alpaca prompt template (`### Instruction:` / `### Response:`, task instruction baked in), `repetition_penalty=1.3` via `make_logits_processors`. Exposes `normalize_text(raw: str) -> str`. Loads the model once at process start (LoRA load takes a few seconds — must not reload per-request).
- `tts_client.py` — `synthesize_speech(text: str, voice: str = "mr-IN-AarohiNeural") -> bytes`, wrapping `edge_tts.Communicate(text, voice).save(...)`. Retry with a short bounded backoff on transient failures (unofficial endpoint — no documented rate limit, but no SLA either, so still worth a retry loop rather than a bare call).
- `server.py` — small FastAPI app (pydantic already a dependency), the thing the UI actually talks to:
  - `POST /api/speak` — body `{text: str}` → `normalize_text()` → `synthesize_speech()` → returns `{normalized_text: str, audio_url: str}` (audio served from a temp/output dir via a static file route, or returned as a base64 audio blob — pick whichever is simpler to wire up first, base64 avoids a second route).
  - Serves `site/index.html` as the root static page — frontend and backend share one origin, so no CORS config needed.
  - Binds to `127.0.0.1` only — this is a single-user local tool, not meant to be reachable from the network. That's the actual justification for skipping auth (see Security), not an oversight.
- `speak.py` — CLI kept as a thin wrapper around the same `normalize_text()` / `synthesize_speech()` functions the server uses, for scripting/eval use without spinning up the server.
- `transcribe.py` — `transcribe(audio_bytes: bytes) -> str` via `mlx-whisper` (Apple-Silicon-native, consistent with the rest of this project's MLX-based tooling; Studio itself already lists `whisper-large-v3` as a supported model, so this isn't introducing an unfamiliar dependency class).
- `eval_wer.py` — iterates `eval.jsonl` (post-cleanup): normalize → synthesize → transcribe → word-level edit-distance WER against the reference `output` field → writes `eval_report.jsonl` (per-row) + prints aggregate WER.
- **Config:** none needed — no API key for `edge-tts`.



### [Security]

- **No auth, by explicit scoped decision, not by oversight:** `server.py` binds to `127.0.0.1` only. Single local user, single machine, never exposed to the network — that's the actual justification. If this is ever deployed anywhere reachable by others, this decision must be revisited and real auth added first.
- **Secrets:** none for the default path — `edge-tts` needs no credential. (If the Sarvam fallback is ever built, its API key follows the exact same `.env`-only rule already used for `GEMINI_API_KEY`.)
- **Input validation, both layers:** the UI disables submission on empty input, but `POST /api/speak` re-validates independently (non-empty, length cap) — client-side checks are a UX nicety, never trusted alone. Reject and return a 400 with a clear message on invalid input.
- **Output encoding in the UI:** the normalized text returned by the model is inserted into the page with `textContent`, never `innerHTML` — the model's output isn't fully controlled input, so treat it like any other untrusted string even in a single-user local tool.
- **Error handling:** `/api/speak` catches normalize/synthesize failures and returns a sanitized `{error: "..."}` message with an appropriate status code — never a raw Python traceback or exception string to the browser.
- **Output file handling:** audio filenames derived from a hash or sequential ID, never from raw user text — Devanagari/punctuation in a filename risks filesystem-illegal characters or path issues, not injection, but still worth avoiding.
- **Reliability, not rate-limiting:** `edge-tts` has no documented quota, but it's an unofficial, unauthenticated endpoint with no SLA — retry transient failures with a short bounded backoff rather than assuming every call succeeds.
- **Logging:** log call outcome (status, latency, retry count) only.
- **No exceptions swallowed on the happy path only:** normalize/synthesize/transcribe failures must surface as a clear error in the UI and CLI, not silent empty outputs (this is exactly the class of bug this project already spent days debugging — a silent wrong-template failure — don't reintroduce it here).



## Implementation Plan

- [ ] Step 1: Add `edge-tts` dependency; confirm `mr-IN-AarohiNeural` voice works with a one-line smoke test
- [ ] Step 2: Write `tts_client.py` with bounded retry
- [ ] Step 3: Write `normalize.py` wrapping the validated `mlx_lm` generate call (load model once, not per-call)
- [ ] Step 4: Write `speak.py` CLI (normalize → synthesize → save audio, sanitized filenames) — proves the pipeline works before wiring a UI to it
- [ ] Step 5: Write `server.py` (FastAPI, `127.0.0.1`-only): `POST /api/speak`, server-side input validation, sanitized error responses, serves `site/index.html`
- [ ] Step 6: Turn `site/index.html` into the live UI — input box, Speak button, loading/error states, `<audio controls>` playback, normalized text shown via `textContent`
- [ ] Step 7: Manual smoke test through the actual browser UI on the known-good sentences already validated earlier (AC-temperature, hotel-bill cases)
- [ ] Step 8: Add `mlx-whisper` dependency; write `transcribe.py`
- [ ] Step 9: Write `eval_wer.py` (batch eval + WER report) — reuses `normalize.py`/`tts_client.py`, independent of the UI
- [ ] Step 10 (only if needed): if `edge-tts` quality/reliability disappoints, add `sarvam_client.py` as a swappable second backend behind the same `synthesize_speech()` interface, using Sarvam's free credit tier — apply the original credential-handling checklist to that module only



## Security Checklist (adapted — single-user local tool, no multi-tenant auth)


| Category       | Check                                                 | Action                                                                       |
| -------------- | ----------------------------------------------------- | ---------------------------------------------------------------------------- |
| Auth           | Server reachable only by the local user?              | Bind `127.0.0.1` only, explicit documented decision                          |
| Secrets        | No credential needed for default path                 | N/A for `edge-tts`; `.env`-only rule applies if Sarvam fallback is added     |
| Input          | Validated server-side, not just client-side           | 400 + clear message on empty/oversized text in `/api/speak`                  |
| Output         | Model text rendered safely in the UI                  | `textContent`, never `innerHTML`                                             |
| Output         | No sensitive data in saved reports/logs               | `eval_report.jsonl` contains only text + metrics, no raw audio bytes/headers |
| Error handling | No raw tracebacks reach the browser                   | `/api/speak` catches and returns sanitized error JSON                        |
| Reliability    | Transient failures don't silently produce empty audio | bounded retry + explicit error on exhaustion                                 |
| Logging        | Failures logged with status/latency                   | never log full response bodies unnecessarily                                 |
| File safety    | Output filenames not derived from raw user text       | hash/sequential ID naming                                                    |


