# Marathi TTS Text Normalization

Off-the-shelf TTS engines read Marathi text with a Hindi accent, a flat monotone, and frequent phonetic mistakes — the retroflex `ळ` collapses into `ल`, times like `२:३०` get read literally instead of spoken naturally, and English loanwords ("Office", "Laptop") come out mangled.

This project fixes that at the text layer: an LLM-based normalizer takes raw, messy Marathi input (mixed digits, English words, no punctuation cues) and rewrites it into clean, phonetically accurate, speech-ready Devanagari — expanded numbers, correct retroflex consonants, transliterated loanwords, and `[pause]` tags marking natural clause breaks. That normalized text is what a downstream TTS engine actually synthesizes, so the resulting audio sounds like a person, not a text-to-speech demo.

```
"मी काल २:३० PM ला ५०० रुपयांचे पुस्तक विकत घेतले, मज्जा आली!"
                        │
                        ▼
        Speech Normalization (this repo)
                        │
                        ▼
"मी काल [pause] दोन वाजून तीस मिनिटांनी [pause] पाचशे रुपयांचे पुस्तक विकत घेतले... [pause] मज्जा आली!"
                        │
                        ▼
              TTS engine → natural Marathi audio
```

## What it handles

| Problem | Typical TTS behavior | This pipeline |
| :--- | :--- | :--- |
| Retroflex `ळ` | Flattened to `ल` (`कमल` instead of `कमळ`) | Preserved correctly (`कमळ`, `बाळ`) |
| Numbers & times | Read digit-by-digit or misread | Spoken-form expansion (`दोन वाजून तीस मिनिटांनी`) |
| English loanwords | Foreign accent, mispronounced | Natural Devanagari transliteration (`ऑफिस`, `लॅपटॉप`) |
| Pauses & rhythm | Flat, no breathing room | `[pause]` tags inserted at natural clause boundaries |
| Conjuncts (`ज्ञा`, `क्ष`) | Often mangled | Rendered accurately |

## How the dataset is built

`normalize_sentence.py` generates the training data for this normalization task using the Gemini API, in two steps:

1. **Generation** — prompts Gemini to produce diverse, natural raw Marathi sentences that mix in numbers, times, percentages, currency, units, and English loanwords, across a range of sentence types (statements, questions, exclamations, commands).
2. **Normalization** — each raw sentence is passed through a second Gemini call with the normalization rules above, producing the phonetically clean, pause-annotated Devanagari version.

Both steps are paced and retried to stay within API rate limits, and every record is flushed to disk as soon as it's produced — interrupting and re-running the script skips whatever's already done and continues.

The result is `marathi_speech_normalization.jsonl`, an Alpaca-style instruction dataset:

```json
{
  "instruction": "Convert the raw Marathi text into phonetically normalized, speech-ready Devanagari text with pause annotations for TTS.",
  "input": "मी काल २:३० PM ला ५०० रुपयांचे पुस्तक विकत घेतले.",
  "output": "मी काल [pause] दोन वाजून तीस मिनिटांनी [pause] पाचशे रुपयांचे पुस्तक विकत घेतले."
}
```

`eda.ipynb` explores the generated dataset — length distributions, `[pause]` tag coverage, sentence-type mix, duplicate checks, and residual-English leak detection — to catch generation issues before the data goes into training.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root with your Gemini API key:

```
GEMINI_API_KEY=your_api_key_here
```

## Usage

Generate (or resume generating) the dataset:

```bash
python3 normalize_sentence.py
```

Explore the dataset:

```bash
jupyter notebook eda.ipynb
```

## Project Vision

This dataset is one piece of a larger three-stage pipeline aimed at human-sounding Marathi TTS:

1. **Speech normalization dataset — implemented (this repo).** Raw Marathi text paired with phonetically normalized, pause-annotated Devanagari output, generated and validated as described above.
2. **Fine-tuned normalization model — planned.** The dataset is used to fine-tune an open-weight instruction model (e.g. Qwen-2.5-7B or Llama-3.1-8B) via QLoRA, so that normalization runs as a fast local model instead of an API call — turning any raw Marathi sentence into speech-ready text on the fly.
3. **TTS synthesis — planned.** The fine-tuned model's normalized output feeds a zero-shot TTS engine (F5-TTS / Kokoro / XTTS) to produce the final `.wav` audio.

```
raw Marathi text → [1: normalization dataset] → [2: fine-tuned LLM] → [3: TTS engine] → audio
                         ✅ done                      ⏳ planned          ⏳ planned
```

The normalization step is the highest-leverage part of this chain: most of the "robotic accent" and mispronunciation complaints against Marathi TTS trace back to bad input text, not a bad voice model. Getting Stage 1 right is what makes Stages 2 and 3 worth building.
