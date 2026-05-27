# Local Whisper

Local-only speech-to-text app powered by Apple's MLX-Whisper. Drag and drop audio/video files
for batch transcription, or click a button to stream from your microphone. All processing
stays on your Mac — nothing is uploaded.

## Requirements

- macOS with Apple Silicon (M1 / M2 / M3 / M4)
- Python 3.10+ (tested on 3.12)

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The first time you pick a model, it will be downloaded from Hugging Face into
`~/.cache/huggingface/hub` (a few hundred MB to ~3 GB depending on size).

## Run

```bash
./run.sh             # just you, on this Mac (http://localhost:8000)
./run.sh --lan       # also reachable from other devices on your Wi-Fi
./run.sh --https     # same as --lan, plus HTTPS (needed for remote mic streaming)
```

### Sharing with someone else on your Wi-Fi

- **`--lan`** binds the server to `0.0.0.0`. On startup it prints a URL like
  `http://192.168.1.42:8000` that you can hand to anyone on your network.
  Drag-and-drop / batch transcription will work fine.
- **`--https`** is required if the *other person* wants live microphone
  transcription. Browsers only allow `getUserMedia` on `localhost` (you) or
  over HTTPS (anyone else). A self-signed certificate is generated once
  into `.certs/`. Visitors will see a "not private" warning the first time
  and need to click through (Safari: "Show Details" → "visit this website";
  Chrome: type `thisisunsafe` on the warning page).
- macOS may prompt the first time asking whether `python3.x` may accept
  incoming network connections — click *Allow*.

## Features

- **Drag and drop** audio or video files (`.mp3`, `.wav`, `.m4a`, `.mp4`, `.mov`, `.mkv`, …).
  Video files have audio extracted automatically.
- **Live streaming** from your microphone, using LocalAgreement-2 to commit stable text and keep
  an "in-progress" tail visible.
- **Model picker** — switch between tiny / base / small / medium / large / turbo on the fly.
  The model is cached after the first load.
- **Export** the transcript as TXT, SRT, or VTT.
- 100% local, no API keys, no network calls after model download.

## How streaming works

The browser captures mic audio with the Web Audio API, downsamples to 16 kHz mono PCM, and
streams it over a WebSocket. The server keeps a rolling buffer (~25 s) and re-runs Whisper
every ~1 s on the buffer. The LocalAgreement-2 policy commits a token only after two
consecutive transcriptions agree on it as a prefix, which keeps the committed text stable
while the tail can keep updating.

## Files

- `app.py` — FastAPI server (REST + WebSocket).
- `whisper_engine.py` — Model loading, batch transcription, streaming buffer.
- `static/` — Web UI (HTML / CSS / JS).
