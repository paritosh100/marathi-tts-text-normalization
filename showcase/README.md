# मराठी आवाज — showcase site

A static, dependency-free showcase site for the Marathi TTS pipeline: two
tabs, no build step, no backend.

- **Listen** — five real sentences run through the pipeline, raw vs.
  normalized text, same IndicF5 voice, audio embedded directly so anyone can
  hear the difference without running any code.
- **Training journey** — real Unsloth Studio and MLflow screenshots, model
  configuration, and the honest current status (including known gaps).

This is separate from `../site/` (the FastAPI-backed live-synthesis demo in
the main repo, which needs the actual model running locally) — this is a
pre-generated, static, host-anywhere version for sharing.

```mermaid
flowchart TD
    subgraph gen ["Local pipeline — run once, not at request time"]
        A["normalize.py"] --> B["indicf5_client.py / tts_client.py"]
        B --> C["10 pre-generated .wav clips"]
    end
    C --> D["showcase/public/"]
    D --> E["showcase/index.html"]
    E --> F["Vercel — static hosting"]
```

Everything in `showcase/` is pre-baked: no server, no live model calls, no
API keys needed at request time — just static files Vercel serves as-is.

## Structure

```
showcase/
  index.html           everything — markup, styles, tab-switching JS
  public/
    audio/              10 pre-generated .wav clips (5 sentences × raw/normalized)
    screenshots/
      unsloth/           5 Unsloth Studio training screenshots
      mlflow/             2 MLflow tracking screenshots
```

## Run locally

No build step — just serve the folder:

```
cd showcase
python3 -m http.server 8080
```

Then open `http://localhost:8080`.

## Deploy to Vercel

Zero config needed — it's a plain static site.

1. Push this repo to GitHub (if not already).
2. In Vercel: **New Project** → import the repo → set **Root Directory** to `showcase`.
3. Framework preset: **Other**. Leave build command and output directory blank.
4. Deploy.

Or via the CLI, from this directory:

```
cd showcase
npx vercel --prod
```
