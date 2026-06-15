"""Whisper engine wrapper: model caching, batch transcription, streaming buffer."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# Use the bundled static ffmpeg so the user doesn't need brew install.
import imageio_ffmpeg
from huggingface_hub import snapshot_download

FFMPEG_BIN = imageio_ffmpeg.get_ffmpeg_exe()

import mlx_whisper  # noqa: E402

log = logging.getLogger("local-whisper.engine")


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

# Curated list of mlx-community models. The repo IDs match Hugging Face.
MODELS: list[dict[str, Any]] = [
    {
        "id": "mlx-community/whisper-tiny-mlx",
        "label": "Tiny (39M, fastest, lowest accuracy)",
        "size_mb": 75,
    },
    {
        "id": "mlx-community/whisper-base-mlx",
        "label": "Base (74M)",
        "size_mb": 145,
    },
    {
        "id": "mlx-community/whisper-small-mlx",
        "label": "Small (244M)",
        "size_mb": 480,
    },
    {
        "id": "mlx-community/whisper-medium-mlx",
        "label": "Medium (769M)",
        "size_mb": 1500,
    },
    {
        "id": "mlx-community/whisper-large-v3-mlx",
        "label": "Large v3 (1.55B, best accuracy)",
        "size_mb": 3100,
    },
    {
        "id": "mlx-community/whisper-large-v3-turbo",
        "label": "Large v3 Turbo (809M, recommended)",
        "size_mb": 1600,
    },
]

DEFAULT_MODEL = "mlx-community/whisper-large-v3-turbo"
SAMPLE_RATE = 16000

# Container extensions that we treat as video (audio gets extracted with ffmpeg).
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".flv", ".wmv", ".mpeg", ".mpg"}


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------


def load_audio(src_path: str | Path) -> np.ndarray:
    """Decode any audio/video file to a float32 mono 16 kHz numpy array.

    We run our bundled ffmpeg directly so mlx-whisper never has to shell out
    to a `ffmpeg` binary on PATH (which may not exist on the user's system).
    """
    src = Path(src_path)
    is_video = src.suffix.lower() in VIDEO_EXTS
    if is_video:
        log.info("extracting audio from video: %s", src.name)
    else:
        log.info("decoding audio: %s", src.name)

    t0 = time.monotonic()
    cmd = [
        FFMPEG_BIN,
        "-nostdin",
        "-threads", "0",
        "-i", str(src_path),
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(SAMPLE_RATE),
        "-",
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        err = result.stderr.decode("utf-8", errors="replace")[-800:]
        raise RuntimeError(f"ffmpeg failed: {err}")
    pcm = np.frombuffer(result.stdout, dtype=np.int16)
    audio = (pcm.astype(np.float32) / 32768.0).copy()
    duration = len(audio) / SAMPLE_RATE
    verb = "extracted" if is_video else "decoded"
    log.info("audio %s: %s of audio in %.2fs", verb, _fmt_duration(duration), time.monotonic() - t0)
    return audio


def is_video(src_path: str | Path) -> bool:
    """True if the file's extension marks it as a video container."""
    return Path(src_path).suffix.lower() in VIDEO_EXTS


def extract_audio(src_path: str | Path, out_path: str | Path) -> Path:
    """Extract the audio track of a media file into a standalone .mp3.

    Uses the bundled ffmpeg, so no system install is required. Returns the
    written output path.
    """
    src, out = Path(src_path), Path(out_path)
    log.info("extracting audio track: %s -> %s", src.name, out.name)
    t0 = time.monotonic()
    cmd = [
        FFMPEG_BIN,
        "-nostdin",
        "-y",
        "-threads", "0",
        "-i", str(src),
        "-vn",                  # drop any video stream
        "-acodec", "libmp3lame",
        "-q:a", "2",            # VBR ~190 kbps
        str(out),
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        err = result.stderr.decode("utf-8", errors="replace")[-800:]
        raise RuntimeError(f"ffmpeg failed: {err}")
    log.info("audio extracted: %s in %.2fs", out.name, time.monotonic() - t0)
    return out


def ensure_model(model_id: str) -> str:
    """Make sure the model weights are on disk; return the local snapshot path.

    First tries a cache-only lookup. If that fails, downloads the repo. Either
    way the path returned can be passed to mlx_whisper.transcribe(path_or_hf_repo=...)
    so it never re-resolves the repo over the network.
    """
    try:
        path = snapshot_download(model_id, local_files_only=True)
        log.info("model cached: %s", model_id)
        return path
    except Exception:
        pass  # not cached -> real download below
    log.info("downloading model: %s (first use; this may take a while)", model_id)
    t0 = time.monotonic()
    path = snapshot_download(model_id)
    log.info("model downloaded: %s in %.1fs", model_id, time.monotonic() - t0)
    return path


def _fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s"


def _parse_ts(ts: str) -> float:
    """Parse mlx-whisper's "HH:MM:SS.mmm" or "MM:SS.mmm" timestamp into seconds."""
    parts = ts.strip().split(":")
    try:
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
    except ValueError:
        pass
    return 0.0


class _SegmentProgress:
    """File-like object that captures mlx-whisper's verbose stdout and turns
    every "[start --> end] text" line into a log line of our own."""

    def __init__(self, total_duration: float):
        self._buf = ""
        self._count = 0
        self._total = max(total_duration, 0.01)
        self._t_first: float | None = None

    def write(self, s: str) -> int:
        if not s:
            return 0
        self._buf += s
        while "\n" in self._buf:
            line, _, rest = self._buf.partition("\n")
            self._buf = rest
            self._emit(line)
        return len(s)

    def flush(self) -> None:
        if self._buf.strip():
            self._emit(self._buf)
            self._buf = ""

    def _emit(self, raw: str) -> None:
        line = raw.strip()
        if not line.startswith("["):
            return  # mlx-whisper occasionally prints non-segment lines (lang detect, etc.) — drop
        try:
            close = line.index("]")
        except ValueError:
            return
        self._count += 1
        if self._t_first is None:
            self._t_first = time.monotonic()
        ts = line[1:close]
        text = line[close + 1:].strip()
        end_str = ts.split("-->")[-1].strip()
        end_secs = _parse_ts(end_str)
        pct = min(100.0, (end_secs / self._total) * 100.0)
        # ETA: extrapolate from elapsed-since-first-segment + processed audio fraction.
        eta_str = ""
        if end_secs > 0 and self._t_first is not None:
            elapsed = time.monotonic() - self._t_first
            if elapsed > 0.5 and pct > 0:
                total_est = elapsed * (100.0 / pct)
                eta = max(0.0, total_est - elapsed)
                eta_str = f", eta {_fmt_duration(eta)}"
        preview = text[:60] + ("…" if len(text) > 60 else "")
        log.info("  seg %4d (%3.0f%% @ %s%s): %s", self._count, pct, end_str, eta_str, preview)


# ---------------------------------------------------------------------------
# Batch transcription
# ---------------------------------------------------------------------------


# mlx-whisper caches loaded models internally by path_or_hf_repo, so we don't
# need our own model cache — we just need a lock to serialize calls.
_inference_lock = threading.Lock()


def transcribe_file(
    file_path: str | Path,
    model_id: str = DEFAULT_MODEL,
    language: str | None = None,
    task: str = "transcribe",  # "transcribe" or "translate"
    word_timestamps: bool = False,
) -> dict[str, Any]:
    """Run a one-shot transcription on a complete file."""
    src = Path(file_path)
    if not src.exists():
        raise FileNotFoundError(src)

    # Decode to a numpy array ourselves — handles audio and video uniformly,
    # and avoids mlx-whisper's internal `ffmpeg` PATH lookup.
    audio = load_audio(src)
    duration = len(audio) / SAMPLE_RATE

    # Make sure the model is on disk (logs cached vs. downloading) and pass the
    # local path so mlx-whisper doesn't redo any HF lookup.
    local_model = ensure_model(model_id)

    kwargs: dict[str, Any] = {
        "path_or_hf_repo": local_model,
        "task": task,
        "word_timestamps": word_timestamps,
        # verbose=True makes mlx-whisper print each segment to stdout as it's
        # decoded; we redirect stdout below and re-emit each line through our
        # logger so the user sees progress in the same stream as everything else.
        "verbose": True,
    }
    if language:
        kwargs["language"] = language

    log.info("transcribing %s of audio with %s ...", _fmt_duration(duration), model_id)
    t0 = time.monotonic()
    progress = _SegmentProgress(duration)
    with _inference_lock, contextlib.redirect_stdout(progress):
        result = mlx_whisper.transcribe(audio, **kwargs)
    progress.flush()
    elapsed = time.monotonic() - t0
    rt = (duration / elapsed) if elapsed > 0 else 0.0
    n_seg = len(result.get("segments", []))
    log.info(
        "transcribed: %d segments in %.2fs (%.1f× realtime, lang=%s)",
        n_seg, elapsed, rt, result.get("language") or "?",
    )

    # Normalize the segments to a JSON-friendly shape.
    segments = [
        {
            "id": int(s.get("id", i)),
            "start": float(s.get("start", 0.0)),
            "end": float(s.get("end", 0.0)),
            "text": s.get("text", "").strip(),
        }
        for i, s in enumerate(result.get("segments", []))
    ]
    return {
        "text": result.get("text", "").strip(),
        "language": result.get("language"),
        "segments": segments,
    }


# ---------------------------------------------------------------------------
# Streaming with LocalAgreement-2
# ---------------------------------------------------------------------------


@dataclass
class StreamingSession:
    """Rolling-buffer streaming transcriber using LocalAgreement-2.

    The client streams 16 kHz mono PCM (float32) chunks. We keep ~25 s of audio
    in memory, re-running Whisper every `tick_seconds`. The LocalAgreement-2
    policy commits a token to the "stable" output only after two consecutive
    transcriptions agree on it as a prefix. The unstable tail is sent as a
    "partial" so the user sees live progress.
    """

    model_id: str = DEFAULT_MODEL
    language: str | None = None
    task: str = "transcribe"

    # Tuning knobs
    buffer_max_seconds: float = 25.0
    trim_to_seconds: float = 12.0  # when buffer exceeds max, trim back to this
    tick_seconds: float = 1.2  # how often to re-run inference
    min_audio_seconds: float = 1.5  # don't bother running on a tiny buffer

    # Internal state
    _audio: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))
    _audio_offset: float = 0.0  # seconds of audio already trimmed from the front
    _prev_words: list[str] = field(default_factory=list)  # tokens from previous inference
    _committed_text: str = ""
    _partial_text: str = ""
    _last_tick: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _closed: bool = False

    # ------- audio -------
    def push_audio(self, samples: np.ndarray) -> None:
        """Append float32 PCM samples (mono, 16 kHz) to the rolling buffer."""
        if samples.dtype != np.float32:
            samples = samples.astype(np.float32)
        with self._lock:
            self._audio = np.concatenate([self._audio, samples])
            # Trim if we've blown past the cap.
            max_samples = int(self.buffer_max_seconds * SAMPLE_RATE)
            if self._audio.shape[0] > max_samples:
                trim_to = int(self.trim_to_seconds * SAMPLE_RATE)
                drop = self._audio.shape[0] - trim_to
                self._audio = self._audio[drop:]
                self._audio_offset += drop / SAMPLE_RATE
                # Reset agreement state — the buffer has shifted under us.
                self._prev_words = []

    def buffer_seconds(self) -> float:
        with self._lock:
            return self._audio.shape[0] / SAMPLE_RATE

    def close(self) -> None:
        self._closed = True

    # ------- inference -------
    def should_tick(self) -> bool:
        if self._closed:
            return False
        if self.buffer_seconds() < self.min_audio_seconds:
            return False
        return (time.monotonic() - self._last_tick) >= self.tick_seconds

    def tick(self) -> dict[str, Any] | None:
        """Run inference on the current buffer. Returns a dict with newly
        committed text and the current partial tail, or None if skipped."""
        if self._closed:
            return None
        with self._lock:
            audio = self._audio.copy()
        if audio.shape[0] < int(self.min_audio_seconds * SAMPLE_RATE):
            return None

        self._last_tick = time.monotonic()

        kwargs: dict[str, Any] = {
            "path_or_hf_repo": self.model_id,
            "task": self.task,
            "verbose": None,
            "condition_on_previous_text": False,
        }
        if self.language:
            kwargs["language"] = self.language

        with _inference_lock:
            try:
                result = mlx_whisper.transcribe(audio, **kwargs)
            except Exception as e:
                return {"error": str(e)}

        text = (result.get("text") or "").strip()
        words = _split_words(text)

        # LocalAgreement-2: commit the longest prefix that matches between this
        # run and the previous run.
        agreed_prefix_len = _longest_common_prefix(self._prev_words, words)
        newly_committed = words[: agreed_prefix_len]
        self._prev_words = words

        new_commit_text = ""
        if newly_committed:
            new_commit_text = _join_words(newly_committed)
            if self._committed_text and not new_commit_text.startswith(" "):
                self._committed_text = (self._committed_text.rstrip() + " " + new_commit_text.lstrip()).strip()
            else:
                self._committed_text = (self._committed_text + " " + new_commit_text).strip()
            # Pop committed words off the front of prev_words so the next tick
            # only looks at the remaining (uncommitted) portion.
            self._prev_words = words[agreed_prefix_len:]

        partial = _join_words(self._prev_words)
        self._partial_text = partial

        return {
            "committed": self._committed_text,
            "partial": partial,
            "new_text": new_commit_text,
            "language": result.get("language"),
        }

    def finalize(self) -> dict[str, Any]:
        """Force-commit whatever partial text remains. Call when stream ends."""
        with self._lock:
            tail = self._partial_text.strip()
        if tail:
            if self._committed_text:
                self._committed_text = (self._committed_text.rstrip() + " " + tail).strip()
            else:
                self._committed_text = tail
        return {"committed": self._committed_text, "partial": ""}


def _split_words(text: str) -> list[str]:
    # Whisper outputs spaces in front of most words; splitting on whitespace
    # and keeping punctuation attached is fine for prefix matching.
    return text.split()


def _join_words(words: list[str]) -> str:
    return " ".join(words)


def _longest_common_prefix(a: list[str], b: list[str]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


# ---------------------------------------------------------------------------
# Async wrappers so the FastAPI event loop never blocks on inference
# ---------------------------------------------------------------------------


async def transcribe_file_async(*args: Any, **kwargs: Any) -> dict[str, Any]:
    return await asyncio.to_thread(transcribe_file, *args, **kwargs)


async def session_tick_async(session: StreamingSession) -> dict[str, Any] | None:
    return await asyncio.to_thread(session.tick)
