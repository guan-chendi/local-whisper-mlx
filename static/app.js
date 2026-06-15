"use strict";

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

const state = {
  // Last completed transcript (for export).
  text: "",
  segments: [],          // [{start, end, text}]
  // Last uploaded video file, kept so its audio can be extracted on demand.
  videoFile: null,
  // Live streaming
  committed: "",
  partial: "",
  streaming: false,
  ws: null,
  audioCtx: null,
  workletNode: null,
  micStream: null,
};

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------

document.addEventListener("DOMContentLoaded", async () => {
  initTabs();
  await loadModels();
  initDropzone();
  initMic();
  initTranscriptControls();
  initAudioDownload();
});

// True when the file is a video container we can extract audio from.
function isVideoFile(file) {
  if (file.type && file.type.startsWith("video/")) return true;
  const m = /\.([a-z0-9]+)$/i.exec(file.name);
  const ext = m ? m[1].toLowerCase() : "";
  return ["mp4", "mov", "mkv", "webm", "avi", "m4v", "flv", "wmv", "mpeg", "mpg"].includes(ext);
}

// ---------------------------------------------------------------------------
// Tabs
// ---------------------------------------------------------------------------

function initTabs() {
  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const tab = btn.dataset.tab;
      document.querySelectorAll(".tab-btn").forEach((b) => b.classList.toggle("active", b === btn));
      document.querySelectorAll(".panel").forEach((p) => {
        p.classList.toggle("active", p.id === `panel-${tab}`);
      });
    });
  });
}

// ---------------------------------------------------------------------------
// Models
// ---------------------------------------------------------------------------

async function loadModels() {
  try {
    const res = await fetch("/api/models");
    const data = await res.json();
    const sel = document.getElementById("model-select");
    sel.innerHTML = "";
    for (const m of data.models) {
      const opt = document.createElement("option");
      opt.value = m.id;
      opt.textContent = m.label;
      if (m.id === data.default) opt.selected = true;
      sel.appendChild(opt);
    }
  } catch (e) {
    console.error("Failed to load models", e);
  }
}

function getOpts() {
  return {
    model: document.getElementById("model-select").value,
    language: document.getElementById("lang-select").value,
    task: document.getElementById("task-select").value,
  };
}

// ---------------------------------------------------------------------------
// Batch / drag-drop
// ---------------------------------------------------------------------------

function initDropzone() {
  const dz = document.getElementById("dropzone");
  const input = document.getElementById("file-input");
  const picker = document.getElementById("pick-file");

  picker.addEventListener("click", (e) => {
    e.stopPropagation();
    input.click();
  });
  dz.addEventListener("click", () => input.click());
  input.addEventListener("change", () => {
    if (input.files && input.files[0]) handleFile(input.files[0]);
    input.value = "";
  });

  // Global drag handlers — accept drops anywhere on the page when batch tab is open.
  ["dragenter", "dragover"].forEach((ev) =>
    dz.addEventListener(ev, (e) => {
      e.preventDefault();
      dz.classList.add("hover");
    })
  );
  ["dragleave", "drop"].forEach((ev) =>
    dz.addEventListener(ev, (e) => {
      e.preventDefault();
      dz.classList.remove("hover");
    })
  );
  dz.addEventListener("drop", (e) => {
    if (e.dataTransfer.files && e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
  });

  // Prevent the browser from navigating away on a misfired drop.
  window.addEventListener("dragover", (e) => e.preventDefault());
  window.addEventListener("drop", (e) => e.preventDefault());
}

async function handleFile(file) {
  if (state.streaming) {
    showBatchStatus("Stop the live recording first.", "error");
    return;
  }
  showBatchStatus(`Transcribing ${file.name} …`, "progress");

  // Offer audio extraction whenever the source is a video.
  state.videoFile = isVideoFile(file) ? file : null;
  showAudioDownload(!!state.videoFile);

  const opts = getOpts();
  const form = new FormData();
  form.append("file", file);
  form.append("model", opts.model);
  form.append("language", opts.language);
  form.append("task", opts.task);

  try {
    const res = await fetch("/api/transcribe", { method: "POST", body: form });
    if (!res.ok) {
      const t = await res.text();
      throw new Error(t || `${res.status} ${res.statusText}`);
    }
    const data = await res.json();
    state.text = data.text || "";
    state.segments = data.segments || [];
    renderSegments();
    enableExports(true);
    showBatchStatus(
      `Done · ${file.name} · ${(state.segments.length || 0)} segments` +
        (data.language ? ` · detected: ${data.language}` : ""),
      "ok"
    );
  } catch (e) {
    console.error(e);
    showBatchStatus(`Failed: ${e.message || e}`, "error");
  }
}

function showBatchStatus(text, kind = "info") {
  const el = document.getElementById("batch-status");
  el.classList.remove("hidden", "error", "ok");
  el.textContent = "";
  const span = document.createElement("span");
  span.textContent = text;
  el.appendChild(span);
  if (kind === "progress") {
    const bar = document.createElement("div");
    bar.className = "progress-bar";
    const fill = document.createElement("div");
    fill.className = "fill";
    bar.appendChild(fill);
    el.appendChild(bar);
  } else if (kind === "error") {
    el.classList.add("error");
  } else if (kind === "ok") {
    el.classList.add("ok");
  }
}

// ---------------------------------------------------------------------------
// Streaming / mic
// ---------------------------------------------------------------------------

function initMic() {
  const btn = document.getElementById("mic-btn");
  btn.addEventListener("click", () => {
    if (state.streaming) stopStreaming();
    else startStreaming();
  });
}

async function startStreaming() {
  if (state.streaming) return;
  setMicStatus("Requesting microphone …");
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
    });
  } catch (e) {
    setMicStatus(`Microphone denied: ${e.message}`, "err");
    return;
  }
  state.micStream = stream;

  const ctx = new (window.AudioContext || window.webkitAudioContext)();
  state.audioCtx = ctx;

  // Inline AudioWorklet processor: pulls 128-sample input frames and forwards
  // them as float32 PCM via the port.
  const workletSrc = `
    class PCMWorklet extends AudioWorkletProcessor {
      process(inputs) {
        const ch = inputs[0][0];
        if (ch && ch.length) this.port.postMessage(ch.slice(0));
        return true;
      }
    }
    registerProcessor('pcm-worklet', PCMWorklet);
  `;
  const blob = new Blob([workletSrc], { type: "application/javascript" });
  const url = URL.createObjectURL(blob);
  try {
    await ctx.audioWorklet.addModule(url);
  } finally {
    URL.revokeObjectURL(url);
  }

  const source = ctx.createMediaStreamSource(stream);
  const node = new AudioWorkletNode(ctx, "pcm-worklet");
  state.workletNode = node;

  // Open the WebSocket.
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  const ws = new WebSocket(`${proto}//${location.host}/ws/stream`);
  ws.binaryType = "arraybuffer";
  state.ws = ws;

  ws.addEventListener("open", () => {
    setWsState("connecting", "on");
    const opts = getOpts();
    ws.send(JSON.stringify({ type: "start", model: opts.model, language: opts.language, task: opts.task }));
  });
  ws.addEventListener("message", (ev) => {
    let msg;
    try { msg = JSON.parse(ev.data); } catch { return; }
    if (msg.type === "ready") {
      setWsState("streaming", "on");
      setMicStatus("Listening …");
    } else if (msg.type === "update") {
      state.committed = msg.committed || "";
      state.partial = msg.partial || "";
      renderLive();
    } else if (msg.type === "final") {
      state.committed = msg.committed || state.committed;
      state.partial = "";
      renderLive();
      // Promote the streamed text to "text" so the user can export.
      state.text = state.committed;
      state.segments = [];
      enableExports(true);
    } else if (msg.type === "error") {
      setMicStatus(`Server error: ${msg.message}`, "err");
    }
  });
  ws.addEventListener("close", () => {
    setWsState("disconnected", "");
    if (state.streaming) stopStreaming();
  });
  ws.addEventListener("error", () => setWsState("error", "err"));

  // Downsample mic audio from ctx.sampleRate -> 16 kHz mono float32 and ship it.
  const inRate = ctx.sampleRate;
  const targetRate = 16000;
  const ratio = inRate / targetRate;
  let leftover = new Float32Array(0);

  const sendChunk = (samples) => {
    if (ws.readyState !== WebSocket.OPEN) return;
    // Float32Array.buffer may be a SharedArrayBuffer in some contexts; copy to be safe.
    const buf = new ArrayBuffer(samples.length * 4);
    new Float32Array(buf).set(samples);
    ws.send(buf);
  };

  node.port.onmessage = (ev) => {
    const incoming = ev.data; // Float32Array @ inRate
    // Update level meter.
    let peak = 0;
    for (let i = 0; i < incoming.length; i++) {
      const v = Math.abs(incoming[i]);
      if (v > peak) peak = v;
    }
    updateLevel(peak);

    // Concat with leftover then downsample.
    const merged = new Float32Array(leftover.length + incoming.length);
    merged.set(leftover);
    merged.set(incoming, leftover.length);

    const outLen = Math.floor(merged.length / ratio);
    if (outLen <= 0) {
      leftover = merged;
      return;
    }
    const out = new Float32Array(outLen);
    for (let i = 0; i < outLen; i++) {
      // Linear interpolation between two source samples.
      const srcPos = i * ratio;
      const i0 = Math.floor(srcPos);
      const i1 = Math.min(i0 + 1, merged.length - 1);
      const frac = srcPos - i0;
      out[i] = merged[i0] * (1 - frac) + merged[i1] * frac;
    }
    const consumed = Math.floor(outLen * ratio);
    leftover = merged.slice(consumed);
    sendChunk(out);
  };

  source.connect(node);
  // Don't pipe to destination — we'd echo the user's own voice back.

  state.streaming = true;
  document.getElementById("mic-btn").classList.add("recording");
  // Clear any prior transcript so the streamed text starts fresh.
  state.committed = "";
  state.partial = "";
  state.text = "";
  state.segments = [];
  enableExports(false);
  renderLive();
}

function stopStreaming() {
  if (!state.streaming) return;
  state.streaming = false;
  document.getElementById("mic-btn").classList.remove("recording");
  setMicStatus("Wrapping up …");
  try { state.ws && state.ws.send(JSON.stringify({ type: "stop" })); } catch {}
  try { state.workletNode && state.workletNode.disconnect(); } catch {}
  if (state.micStream) {
    state.micStream.getTracks().forEach((t) => t.stop());
    state.micStream = null;
  }
  if (state.audioCtx) {
    state.audioCtx.close().catch(() => {});
    state.audioCtx = null;
  }
  // Allow the server a moment to emit `final` before we drop the socket.
  setTimeout(() => {
    try { state.ws && state.ws.close(); } catch {}
    state.ws = null;
    setMicStatus("Click to start streaming");
    setWsState("disconnected", "");
    updateLevel(0);
  }, 400);
}

function setMicStatus(text, dotState) {
  document.getElementById("mic-status").textContent = text;
  if (dotState !== undefined) setWsState(undefined, dotState);
}

function setWsState(text, dotClass) {
  if (text !== undefined) document.getElementById("ws-state").textContent = text;
  const dot = document.getElementById("ws-dot");
  if (dotClass !== undefined) {
    dot.classList.remove("on", "err");
    if (dotClass) dot.classList.add(dotClass);
  }
}

function updateLevel(peak) {
  const pct = Math.min(100, Math.round(peak * 140));
  document.getElementById("level-bar").style.width = `${pct}%`;
}

// ---------------------------------------------------------------------------
// Transcript rendering
// ---------------------------------------------------------------------------

function renderSegments() {
  const el = document.getElementById("transcript");
  el.innerHTML = "";
  if (!state.segments.length) {
    if (state.text) {
      el.textContent = state.text;
    } else {
      const p = document.createElement("span");
      p.className = "transcript-placeholder";
      p.textContent = "Nothing yet.";
      el.appendChild(p);
    }
    return;
  }
  for (const s of state.segments) {
    const line = document.createElement("div");
    line.className = "segment-line";
    const t = document.createElement("span");
    t.className = "segment-time";
    t.textContent = formatTime(s.start);
    line.appendChild(t);
    line.appendChild(document.createTextNode(s.text));
    el.appendChild(line);
  }
}

function renderLive() {
  const el = document.getElementById("transcript");
  el.innerHTML = "";
  if (!state.committed && !state.partial) {
    const p = document.createElement("span");
    p.className = "transcript-placeholder";
    p.textContent = "Listening …";
    el.appendChild(p);
    return;
  }
  if (state.committed) {
    el.appendChild(document.createTextNode(state.committed));
  }
  if (state.partial) {
    if (state.committed) el.appendChild(document.createTextNode(" "));
    const span = document.createElement("span");
    span.className = "partial";
    span.textContent = state.partial;
    el.appendChild(span);
  }
  // Auto-scroll to the bottom for live mode.
  el.scrollTop = el.scrollHeight;
}

function formatTime(t) {
  if (!isFinite(t)) return "00:00";
  const total = Math.floor(t);
  const m = Math.floor(total / 60).toString().padStart(2, "0");
  const s = (total % 60).toString().padStart(2, "0");
  return `${m}:${s}`;
}

// ---------------------------------------------------------------------------
// Transcript actions
// ---------------------------------------------------------------------------

function initTranscriptControls() {
  document.querySelectorAll(".export-btn").forEach((btn) => {
    btn.addEventListener("click", async () => {
      if (btn.id === "clear-btn") {
        state.text = "";
        state.segments = [];
        state.committed = "";
        state.partial = "";
        state.videoFile = null;
        showAudioDownload(false);
        renderSegments();
        enableExports(false);
        return;
      }
      const fmt = btn.dataset.fmt;
      const action = btn.dataset.action;
      const payload = { text: state.text || state.committed, segments: state.segments };
      try {
        const res = await fetch(`/api/export/${fmt}`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        if (!res.ok) throw new Error(await res.text());
        const body = await res.text();
        if (action === "download") {
          downloadFile(`transcript.${fmt}`, body);
        } else {
          await navigator.clipboard.writeText(body);
          flashButton(btn, "Copied!");
        }
      } catch (e) {
        flashButton(btn, "Failed");
        console.error(e);
      }
    });
  });
}

// ---------------------------------------------------------------------------
// Audio extraction (video → mp3)
// ---------------------------------------------------------------------------

function showAudioDownload(visible) {
  document.getElementById("batch-actions").classList.toggle("hidden", !visible);
}

function initAudioDownload() {
  const btn = document.getElementById("download-audio-btn");
  btn.addEventListener("click", async () => {
    const file = state.videoFile;
    if (!file) return;
    const orig = btn.textContent;
    btn.disabled = true;
    btn.textContent = "Extracting audio …";
    try {
      const form = new FormData();
      form.append("file", file);
      const res = await fetch("/api/extract-audio", { method: "POST", body: form });
      if (!res.ok) throw new Error((await res.text()) || `${res.status} ${res.statusText}`);
      const blob = await res.blob();
      const name = file.name.replace(/\.[^.]+$/, "") + ".mp3";
      downloadBlob(name, blob);
      btn.textContent = "Saved ✓";
      setTimeout(() => (btn.textContent = orig), 1500);
    } catch (e) {
      console.error(e);
      btn.textContent = "Failed";
      setTimeout(() => (btn.textContent = orig), 1500);
    } finally {
      btn.disabled = false;
    }
  });
}

function enableExports(enabled) {
  document.querySelectorAll(".export-btn").forEach((b) => {
    if (b.id === "clear-btn") return;
    b.disabled = !enabled;
  });
}

function downloadFile(name, body) {
  downloadBlob(name, new Blob([body], { type: "text/plain;charset=utf-8" }));
}

function downloadBlob(name, blob) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url; a.download = name;
  document.body.appendChild(a); a.click(); a.remove();
  URL.revokeObjectURL(url);
}

function flashButton(btn, text) {
  const orig = btn.textContent;
  btn.textContent = text;
  setTimeout(() => (btn.textContent = orig), 1200);
}
