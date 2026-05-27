"""Whisper engine wrapper: model caching, batch transcription, streaming buffer."""

from __future__ import annotations

import asyncio
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# Use the bundled static ffmpeg so the user doesn't need brew install.
import imageio_ffmpeg

FFMPEG_BIN = imageio_ffmpeg.get_ffmpeg_exe()

import mlx_whisper  # noqa: E402


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
    return (pcm.astype(np.float32) / 32768.0).copy()


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

    kwargs: dict[str, Any] = {
        "path_or_hf_repo": model_id,
        "task": task,
        "word_timestamps": word_timestamps,
        "verbose": None,  # suppress mlx-whisper's stdout chatter
    }
    if language:
        kwargs["language"] = language

    with _inference_lock:
        result = mlx_whisper.transcribe(audio, **kwargs)

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
