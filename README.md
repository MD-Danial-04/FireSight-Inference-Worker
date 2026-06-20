# FireSight Inference Worker

FastAPI worker for SCDF fire report field extraction and audio transcription.

## Setup

Requires **Python 3.12** (pinned pydantic wheels may not install on 3.14 yet).

```bash
cd firesight-nim-worker
python3.12 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

For real transcription, install **ffmpeg** on the host (required for audio decoding):

```bash
# Arch Linux
sudo pacman -S ffmpeg
```

## Run

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## Phase 1 — fake extraction (default)

Health check:

```bash
curl http://localhost:8000/health
```

Fake extraction:

```bash
curl -X POST http://localhost:8000/v1/extract \
  -H "Content-Type: application/json" \
  -d '{
    "text": "LF812 stop for location at 7 Gul Ave. Case classified as False alarm malfunction of manual Call point, at zone 7. Upon investigation No smoke no fire. Case handed over to S3 alsyraf waT190350 from Nanyang NPC",
    "type": "stop_message",
    "incident_type_name": "False Alarm Malfunction"
  }'
```

Expected: JSON with `fields`, `confidence`, and `"source": "fake"`.

Fake transcription:

```bash
curl -X POST http://localhost:8000/v1/transcribe \
  -F "file=@sample.webm;type=audio/webm"
```

Expected: JSON with a static transcript and `"source": "fake"`.

## Phase 2 — local Whisper transcription

Set in `.env`:

```text
USE_FAKE_TRANSCRIPTION=false
WHISPER_MODEL=base
WHISPER_DEVICE=cuda
WHISPER_COMPUTE_TYPE=float16
WHISPER_LANGUAGE=en
```

Restart the worker. First startup downloads the model (~150 MB for `base`) and loads it onto the GPU.

Transcribe an audio file:

```bash
curl -X POST http://localhost:8000/v1/transcribe \
  -F "file=@sample.webm;type=audio/webm"
```

Expected: JSON with `"source": "whisper"` and a non-empty `transcript`.

### Model size tradeoffs

| Model | VRAM (approx) | Notes |
|-------|---------------|-------|
| `tiny` | ~1 GB | Fastest, lower accuracy |
| `base` | ~1 GB | Good default for dev/laptop |
| `small` | ~2 GB | **Recommended on work PC (GB10)** |
| `medium` | ~5 GB | Better accuracy for officer/NPC names |

Use `WHISPER_COMPUTE_TYPE=int8` and `WHISPER_DEVICE=cpu` if CUDA is unavailable.

Optional domain hint for better SCDF jargon recognition:

```text
WHISPER_INITIAL_PROMPT=SCDF fire incident stop message. Appliance call sign, location, zone, classification, handover officer, NPC.
```

## Phase 3 — Ollama field extraction (work PC)

Real extraction uses **Ollama** (`llama3.1:8b`) via an OpenAI-compatible API. No Docker required — run natively on the work PC (ARM64 GB10: build and run on that machine, not cross-build from x86).

### Terminal 1 — start Ollama

```bash
ollama pull llama3.1:8b
ollama serve
```

### Terminal 2 — start worker

Copy this repo to the work PC, then:

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Uncomment the production block in `.env` (or set manually):

```text
USE_FAKE_EXTRACTION=false
USE_FAKE_TRANSCRIPTION=false
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL=llama3.1:8b
LLM_API_KEY=ollama
WHISPER_MODEL=small
WHISPER_DEVICE=cuda
WHISPER_COMPUTE_TYPE=float16
WHISPER_INITIAL_PROMPT=SCDF fire incident stop message. Appliance call sign, location, zone, classification, handover officer, NPC.
```

Start the worker:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Smoke tests

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/v1/extract \
  -H "Content-Type: application/json" \
  -d '{
    "text": "LF812 stop for location at 7 Gul Ave. Case classified as False alarm malfunction of manual Call point, at zone 7.",
    "type": "stop_message",
    "incident_type_name": "False Alarm Malfunction"
  }'
```

Expected: `"source": "ollama"` with extracted fields.

Full pipeline (audio → transcript → fields):

```bash
TRANSCRIPT=$(curl -s -X POST http://localhost:8000/v1/transcribe -F "file=@sample.webm" | jq -r .transcript)
curl -X POST http://localhost:8000/v1/extract \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"$TRANSCRIPT\", \"type\": \"stop_message\"}"
```

If Whisper CUDA fails on GB10 ARM, fallback:

```text
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8
```

### Deprecated env vars

`NIM_LLM_BASE_URL`, `NIM_LLM_MODEL`, and `NIM_API_KEY` are still accepted as aliases for the `LLM_*` settings.

## Tests

```bash
pytest -q
```

Tests run with fake extraction and transcription enabled (no GPU, Ollama, or model download required).

## Later phases

| Phase | Action |
|-------|--------|
| 4 | Dockerize on work PC, OpenShift proxy, Vercel wiring |
