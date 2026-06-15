"""FastAPI server for Local Whisper."""

from __future__ import annotations

import asyncio
import json
import logging
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTask

from whisper_engine import (
    DEFAULT_MODEL,
    MODELS,
    SAMPLE_RATE,
    StreamingSession,
    ensure_model,
    extract_audio,
    is_video,
    session_tick_async,
    transcribe_file_async,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("local-whisper")

ROOT = Path(__file__).parent
UPLOAD_DIR = ROOT / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
STATIC_DIR = ROOT / "static"

app = FastAPI(title="Local Whisper")


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------


@app.get("/api/models")
def list_models() -> dict[str, Any]:
    return {"models": MODELS, "default": DEFAULT_MODEL}


@app.post("/api/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    model: str = Form(DEFAULT_MODEL),
    language: str = Form(""),
    task: str = Form("transcribe"),
) -> JSONResponse:
    """Batch-transcribe an uploaded audio or video file."""
    if not file.filename:
        raise HTTPException(400, "Missing filename")

    suffix = Path(file.filename).suffix or ".bin"
    tmp_path = UPLOAD_DIR / f"{uuid.uuid4().hex}{suffix}"
    t_request = time.monotonic()
    log.info("=== batch request: %s (model=%s, lang=%s) ===", file.filename, model, language or "auto")

    try:
        with tmp_path.open("wb") as f:
            while True:
                chunk = await file.read(1 << 20)  # 1 MB
                if not chunk:
                    break
                f.write(chunk)
        log.info("upload received: %s (%.1f MB)", file.filename, tmp_path.stat().st_size / (1 << 20))

        result = await transcribe_file_async(
            tmp_path,
            model_id=model,
            language=language or None,
            task=task,
        )
        result["filename"] = file.filename
        log.info("=== batch done: %s in %.2fs total ===", file.filename, time.monotonic() - t_request)
        return JSONResponse(result)
    except Exception as e:
        log.exception("Batch transcription failed")
        raise HTTPException(500, str(e))
    finally:
        tmp_path.unlink(missing_ok=True)


@app.post("/api/extract-audio")
async def extract_audio_endpoint(file: UploadFile = File(...)) -> FileResponse:
    """Extract the audio track from an uploaded video and return it as .mp3."""
    if not file.filename:
        raise HTTPException(400, "Missing filename")
    if not is_video(file.filename):
        raise HTTPException(400, "Audio extraction is only available for video files")

    suffix = Path(file.filename).suffix or ".bin"
    token = uuid.uuid4().hex
    src_path = UPLOAD_DIR / f"{token}{suffix}"
    out_path = UPLOAD_DIR / f"{token}.mp3"
    log.info("=== extract-audio request: %s ===", file.filename)

    try:
        with src_path.open("wb") as f:
            while True:
                chunk = await file.read(1 << 20)  # 1 MB
                if not chunk:
                    break
                f.write(chunk)
        await asyncio.to_thread(extract_audio, src_path, out_path)
    except Exception as e:
        src_path.unlink(missing_ok=True)
        out_path.unlink(missing_ok=True)
        log.exception("Audio extraction failed")
        raise HTTPException(500, str(e))
    finally:
        # The source video is no longer needed once extraction has run (or failed).
        src_path.unlink(missing_ok=True)

    download_name = f"{Path(file.filename).stem}.mp3"

    def _cleanup() -> None:
        out_path.unlink(missing_ok=True)

    return FileResponse(
        str(out_path),
        media_type="audio/mpeg",
        filename=download_name,
        background=BackgroundTask(_cleanup),
    )


@app.post("/api/export/{fmt}")
async def export(fmt: str, payload: dict[str, Any]) -> PlainTextResponse:
    """Convert a transcript payload to txt / srt / vtt."""
    segments = payload.get("segments") or []
    text = payload.get("text", "")
    fmt = fmt.lower()
    if fmt == "txt":
        body = text or "\n".join(s["text"] for s in segments)
        return PlainTextResponse(body, media_type="text/plain")
    if fmt == "srt":
        return PlainTextResponse(_to_srt(segments), media_type="text/plain")
    if fmt == "vtt":
        return PlainTextResponse(_to_vtt(segments), media_type="text/plain")
    raise HTTPException(400, f"Unknown format: {fmt}")


def _fmt_time(t: float, sep: str) -> str:
    if t < 0:
        t = 0
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def _to_srt(segments: list[dict[str, Any]]) -> str:
    out = []
    for i, s in enumerate(segments, 1):
        out.append(str(i))
        out.append(f"{_fmt_time(s['start'], ',')} --> {_fmt_time(s['end'], ',')}")
        out.append(s["text"].strip())
        out.append("")
    return "\n".join(out)


def _to_vtt(segments: list[dict[str, Any]]) -> str:
    out = ["WEBVTT", ""]
    for s in segments:
        out.append(f"{_fmt_time(s['start'], '.')} --> {_fmt_time(s['end'], '.')}")
        out.append(s["text"].strip())
        out.append("")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# WebSocket streaming
# ---------------------------------------------------------------------------


@app.websocket("/ws/stream")
async def ws_stream(ws: WebSocket) -> None:
    """Real-time mic streaming.

    Client → server:
      - First message (text/JSON): { "type": "start", "model": "...", "language": "...", "task": "..." }
      - Subsequent (binary): raw float32 PCM @ 16 kHz mono
      - Optional text/JSON: { "type": "stop" }

    Server → client (text/JSON):
      - { "type": "ready" }
      - { "type": "update", "committed": "...", "partial": "...", "new_text": "..." }
      - { "type": "final", "committed": "..." }
      - { "type": "error", "message": "..." }
    """
    await ws.accept()
    session: StreamingSession | None = None
    ticker_task: asyncio.Task[None] | None = None
    stop_event = asyncio.Event()

    async def ticker() -> None:
        """Background loop that runs inference on the rolling buffer."""
        try:
            while not stop_event.is_set():
                if session and session.should_tick():
                    update = await session_tick_async(session)
                    if update is None:
                        pass
                    elif "error" in update:
                        await ws.send_text(json.dumps({"type": "error", "message": update["error"]}))
                    else:
                        await ws.send_text(json.dumps({"type": "update", **update}))
                await asyncio.sleep(0.15)
        except Exception:
            log.exception("ticker crashed")

    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break

            if "text" in msg and msg["text"] is not None:
                try:
                    data = json.loads(msg["text"])
                except json.JSONDecodeError:
                    continue
                kind = data.get("type")
                if kind == "start":
                    if session is None:
                        model_id = data.get("model") or DEFAULT_MODEL
                        log.info("=== stream request (model=%s, lang=%s) ===", model_id, data.get("language") or "auto")
                        # Pre-resolve the model so the first tick isn't blocked on a download.
                        await asyncio.to_thread(ensure_model, model_id)
                        session = StreamingSession(
                            model_id=model_id,
                            language=data.get("language") or None,
                            task=data.get("task") or "transcribe",
                        )
                        ticker_task = asyncio.create_task(ticker())
                        await ws.send_text(json.dumps({"type": "ready"}))
                        log.info("stream ready; listening")
                elif kind == "stop":
                    break

            elif "bytes" in msg and msg["bytes"] is not None and session is not None:
                # Incoming binary frame = float32 PCM little-endian.
                samples = np.frombuffer(msg["bytes"], dtype=np.float32)
                session.push_audio(samples)

    except WebSocketDisconnect:
        pass
    except Exception:
        log.exception("ws_stream crashed")
        try:
            await ws.send_text(json.dumps({"type": "error", "message": "Internal server error"}))
        except Exception:
            pass
    finally:
        stop_event.set()
        if ticker_task is not None:
            try:
                await asyncio.wait_for(ticker_task, timeout=2.0)
            except (asyncio.TimeoutError, Exception):
                ticker_task.cancel()
        if session is not None:
            # One last tick + force-commit any partial tail.
            final = session.finalize()
            try:
                await ws.send_text(json.dumps({"type": "final", **final}))
            except Exception:
                pass
            session.close()
        try:
            await ws.close()
        except Exception:
            pass
        log.info("=== stream ended ===")


# ---------------------------------------------------------------------------
# Static UI
# ---------------------------------------------------------------------------


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "index.html"))
