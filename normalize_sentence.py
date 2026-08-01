import json
import time
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai import errors

# 1. Initialize Gemini Client (Uses GEMINI_API_KEY from .env)
load_dotenv()
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# Free tier for gemini-3.5-flash-lite allows 15 requests/minute per project.
MIN_SECONDS_BETWEEN_CALLS = 4.5
MAX_429_RETRIES = 5
_last_call_at = [0.0]


def call_gemini(model: str, prompt: str, system_instruction: str, temperature: float):
    wait = MIN_SECONDS_BETWEEN_CALLS - (time.monotonic() - _last_call_at[0])
    if wait > 0:
        time.sleep(wait)
    for attempt in range(MAX_429_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    response_mime_type="application/json",
                    temperature=temperature,
                ),
            )
            _last_call_at[0] = time.monotonic()
            return response
        except (errors.ClientError, errors.ServerError) as e:
            _last_call_at[0] = time.monotonic()
            retryable = e.code == 429 or isinstance(e, errors.ServerError)
            if retryable and attempt < MAX_429_RETRIES:
                backoff = MIN_SECONDS_BETWEEN_CALLS * (attempt + 1)
                print(f"{e.code} {e.status}, retrying in {backoff:.1f}s...")
                time.sleep(backoff)
                continue
            raise

# System prompt defining the exact speech-normalization rules for Marathi TTS
SYSTEM_PROMPT = """
You are an expert Marathi linguist and Text-to-Speech (TTS) normalization engine.
Your task is to take raw Marathi text and convert it into phonetically clear, speech-ready Devanagari text.

Rules:
1. Expand all numbers into full Marathi spoken words (e.g., '२:३०' -> 'दोन वाजून तीस मिनिटांनी', '५००' -> 'पाचशे', '१०km' -> 'दहा किलोमीटर').
2. Expand symbols and abbreviations (e.g., '%' -> 'टक्के', 'rs' -> 'रुपये', 'p.m.' -> 'दुपारी').
3. Convert English words into accurate Devanagari phonetics (e.g., 'Office' -> 'ऑफिस', 'Ticket' -> 'तिकीट', 'Laptop' -> 'लॅपटॉप').
4. Insert '[pause]' tags where natural human breathing or pauses occur at clause boundaries or punctuation marks.
5. Ensure proper retroflex consonant 'ळ' and conjuncts ('ज्ञा', 'क्ष') are rendered accurately in Devanagari.

Return ONLY a JSON object with a single key "normalized_text".
"""

GENERATION_SYSTEM_PROMPT = """
You are generating raw (un-normalized) Marathi sentences for a TTS text-normalization dataset.
Each sentence must be natural, everyday Marathi and MUST mix in some of: digits/numbers,
clock times (e.g. २:३०), percentages, currency (rs/₹), units (km/kg/GB), English loanwords
(Office, Ticket, Laptop, etc.), and varied punctuation. Vary the topic and sentence length.
Also vary the sentence type/expression across the batch: plain statements, questions (प्रश्नार्थी),
exclamations (उद्गारार्थी), commands/requests (आज्ञार्थी), and negations — not just declarative statements.
Return ONLY a JSON object with a single key "sentences" containing an array of strings.
"""

TARGET_COUNT = 2000
GENERATION_BATCH_SIZE = 25
MAX_GENERATION_ATTEMPTS = TARGET_COUNT // GENERATION_BATCH_SIZE * 4  # safety cap against stalled dedup


def merge_unique(existing: set, new_items: list) -> set:
    existing.update(s.strip() for s in new_items if s and s.strip())
    return existing


def generate_raw_sentences(batch_size: int) -> list:
    prompt = f"Generate {batch_size} raw Marathi sentences."
    response = call_gemini("gemini-3.5-flash-lite", prompt, GENERATION_SYSTEM_PROMPT, temperature=0.9)
    res_json = json.loads(response.text)
    return res_json.get("sentences", [])


def build_raw_sentences(target_count: int = TARGET_COUNT) -> list:
    pool = set()
    attempts = 0
    while len(pool) < target_count and attempts < MAX_GENERATION_ATTEMPTS:
        batch = generate_raw_sentences(GENERATION_BATCH_SIZE)
        pool = merge_unique(pool, batch)
        attempts += 1
        print(f"Generated {len(pool)}/{target_count} unique raw sentences...")
    return list(pool)[:target_count]

def normalize_sentence(raw_text: str) -> str:
    prompt = f"Raw Marathi Text: {raw_text}"
    response = call_gemini("gemini-3.5-flash-lite", prompt, SYSTEM_PROMPT, temperature=0.2)
    res_json = json.loads(response.text)
    return res_json.get("normalized_text", raw_text)

def main():
    output_filename = "marathi_speech_normalization.jsonl"

    already_done = set()
    if os.path.exists(output_filename):
        with open(output_filename, "r", encoding="utf-8") as f:
            already_done = {json.loads(line)["input"] for line in f if line.strip()}

    remaining = max(0, TARGET_COUNT - len(already_done))
    raw_sentences = build_raw_sentences(remaining) if remaining else []
    todo = [r for r in raw_sentences if r not in already_done]
    print(f"Processing {len(todo)} sentences ({len(already_done)} already done)...")

    written = 0
    with open(output_filename, "a", encoding="utf-8") as f:
        for idx, raw in enumerate(todo, 1):
            try:
                normalized = normalize_sentence(raw)
                entry = {
                    "instruction": "Convert the raw Marathi text into phonetically normalized, speech-ready Devanagari text with pause annotations for TTS.",
                    "input": raw,
                    "output": normalized
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                f.flush()
                written += 1
                print(f"[{idx}/{len(todo)}] Success:")
                print(f"  Input : {raw}")
                print(f"  Output: {normalized}\n")
            except Exception as e:
                print(f"Error processing item {idx}: {e}")

    print(f"🎉 Wrote {written} new records ({len(already_done) + written} total) to '{output_filename}'!")

def _demo():
    pool = merge_unique(set(), ["a", "b", " ", "a", "  c  "])
    assert pool == {"a", "b", "c"}, pool
    pool = merge_unique(pool, ["c", "d"])
    assert pool == {"a", "b", "c", "d"}, pool
    print("merge_unique self-check passed")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--selfcheck":
        _demo()
    else:
        main()