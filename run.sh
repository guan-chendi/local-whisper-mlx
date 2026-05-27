#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  echo "Creating virtualenv..."
  python3 -m venv .venv
fi
source .venv/bin/activate

if ! python -c "import mlx_whisper" 2>/dev/null; then
  echo "Installing dependencies..."
  pip install -r requirements.txt
fi

LAN=0
HTTPS=0
for arg in "$@"; do
  case "$arg" in
    --lan) LAN=1 ;;
    --https) HTTPS=1; LAN=1 ;;  # serving HTTPS without LAN exposure is pointless
    -h|--help)
      cat <<EOF
Usage: ./run.sh [--lan] [--https]

  (no flags)   Bind to 127.0.0.1 only. Just you.
  --lan        Bind to 0.0.0.0 so other devices on your Wi-Fi can connect.
               Batch (drag-drop) works over plain HTTP.
               Microphone streaming will NOT work on remote browsers without --https.
  --https      Serve over HTTPS with an auto-generated self-signed cert.
               Required for microphone streaming from any device that isn't this one.
               Visitors will see a "not private" warning once and need to click through.

Env vars: HOST, PORT (defaults: 127.0.0.1 / 0.0.0.0, 8000)
EOF
      exit 0
      ;;
  esac
done

PORT="${PORT:-8000}"
if [ "$LAN" = 1 ]; then
  HOST="${HOST:-0.0.0.0}"
else
  HOST="${HOST:-127.0.0.1}"
fi

# Find the LAN IP for display. Tries common macOS interfaces.
LAN_IP=""
for IF in en0 en1 en2 en3; do
  IP="$(ipconfig getifaddr "$IF" 2>/dev/null || true)"
  if [ -n "$IP" ]; then LAN_IP="$IP"; break; fi
done

SSL_ARGS=()
SCHEME="http"
if [ "$HTTPS" = 1 ]; then
  mkdir -p .certs
  if [ ! -f .certs/key.pem ] || [ ! -f .certs/cert.pem ]; then
    echo "Generating self-signed certificate (one-time)..."
    SAN="DNS:localhost,IP:127.0.0.1"
    if [ -n "$LAN_IP" ]; then SAN="$SAN,IP:$LAN_IP"; fi
    openssl req -x509 -newkey rsa:2048 -nodes -days 3650 \
      -keyout .certs/key.pem -out .certs/cert.pem \
      -subj "/CN=local-whisper" \
      -addext "subjectAltName=$SAN" >/dev/null 2>&1
  fi
  SSL_ARGS=(--ssl-keyfile .certs/key.pem --ssl-certfile .certs/cert.pem)
  SCHEME="https"
fi

echo ""
echo "  Local Whisper running:"
echo "    ${SCHEME}://localhost:${PORT}"
if [ "$LAN" = 1 ] && [ -n "$LAN_IP" ]; then
  echo "    ${SCHEME}://${LAN_IP}:${PORT}    <- share this with others on your Wi-Fi"
fi
if [ "$LAN" = 1 ] && [ "$HTTPS" = 0 ]; then
  echo ""
  echo "  Note: --lan alone is HTTP. Microphone streaming will not work on remote"
  echo "        browsers. Use --https if your wife wants live mic transcription."
fi
echo ""
echo "  Press Ctrl+C to stop."
echo ""

exec uvicorn app:app --host "$HOST" --port "$PORT" "${SSL_ARGS[@]}"
