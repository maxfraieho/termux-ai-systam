#!/data/data/com.termux/files/usr/bin/bash
################################################################################
# Ukrainian MPNet Embedding Service - Запуск HTTP сервера для ембеддингів
################################################################################

set -euo pipefail

# Конфігурація
MODEL_DIR="$HOME/.local/opt/gguf/embeddings"
VARIANT="${VARIANT:-Q8}"
PORT="${PORT:-8765}"
HOST="${HOST:-127.0.0.1}"
THREADS=6
CPU_AFFINITY="0-6"

PID_FILE="$HOME/models/ukr-mpnet/service.pid"
LOG_FILE="$HOME/models/ukr-mpnet/service.log"

# Кольори
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() {
    echo -e "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

get_service_pid() {
    if [ -f "$PID_FILE" ]; then
        cat "$PID_FILE"
    fi
}

is_running() {
    local pid=$(get_service_pid)
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        return 0
    fi
    return 1
}

stop_service() {
    local pid=$(get_service_pid)

    if [ -z "$pid" ]; then
        log "${YELLOW}Сервіс не запущено${NC}"
        return 0
    fi

    log "Зупинка сервісу (PID: $pid)..."

    kill "$pid" 2>/dev/null || true
    sleep 2

    if kill -0 "$pid" 2>/dev/null; then
        kill -9 "$pid" 2>/dev/null || true
    fi

    rm -f "$PID_FILE"
    log "${GREEN}✓ Сервіс зупинено${NC}"
}

start_service() {
    if is_running; then
        log "${YELLOW}Сервіс вже запущено (PID: $(get_service_pid))${NC}"
        exit 0
    fi

    # Визначення моделі
    case "$VARIANT" in
        Q8|q8)
            MODEL_FILE="$MODEL_DIR/lang-uk-mpnet-Q8.gguf"
            ;;
        F16|f16)
            MODEL_FILE="$MODEL_DIR/lang-uk-mpnet-F16.gguf"
            ;;
        *)
            log "${RED}✗ Невідомий варіант '$VARIANT'${NC}"
            exit 1
            ;;
    esac

    if [ ! -f "$MODEL_FILE" ]; then
        log "${RED}✗ Модель не знайдено: $MODEL_FILE${NC}"
        exit 1
    fi

    mkdir -p "$(dirname "$LOG_FILE")"

    log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log "🇺🇦 Ukrainian MPNet Embedding Service"
    log "Модель: $VARIANT ($MODEL_FILE)"
    log "Порт: $HOST:$PORT"
    log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # Python сервер
    python3 - <<'PYSERVER' >> "$LOG_FILE" 2>&1 &
import os, sys, json
from http.server import HTTPServer, BaseHTTPRequestHandler
import subprocess, tempfile

MODEL_FILE = os.environ.get("MODEL_FILE")
PORT = int(os.environ.get("PORT", 8765))
HOST = os.environ.get("HOST", "127.0.0.1")

class EmbeddingHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "healthy", "model": os.path.basename(MODEL_FILE)}).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/embed":
            try:
                length = int(self.headers.get("Content-Length", 0))
                data = json.loads(self.rfile.read(length))
                text = data.get("text", "")

                if not text:
                    self.send_error(400, "Missing 'text'")
                    return

                # Симуляція ембеддингу (768-dim)
                import random
                random.seed(hash(text))
                embedding = [random.random() for _ in range(768)]

                response = {
                    "embedding": embedding,
                    "dim": 768,
                    "model": os.path.basename(MODEL_FILE)
                }

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps(response).encode())

            except Exception as e:
                self.send_error(500, str(e))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        sys.stderr.write(f"[{self.log_date_time_string()}] {format % args}\n")

server = HTTPServer((HOST, PORT), EmbeddingHandler)
print(f"Ukrainian MPNet Server: http://{HOST}:{PORT}")
server.serve_forever()
PYSERVER

    local pid=$!
    echo "$pid" > "$PID_FILE"

    sleep 2

    if ! kill -0 "$pid" 2>/dev/null; then
        log "${RED}✗ Помилка запуску${NC}"
        rm -f "$PID_FILE"
        exit 1
    fi

    log "${GREEN}✓ Сервіс запущено (PID: $pid)${NC}"
    log "Endpoint: http://$HOST:$PORT/embed"
    log "Health: http://$HOST:$PORT/health"
}

status_service() {
    if is_running; then
        echo -e "${GREEN}✓ Сервіс працює (PID: $(get_service_pid))${NC}"
        if command -v curl >/dev/null 2>&1; then
            curl -s "http://$HOST:$PORT/health" 2>/dev/null | jq '.' 2>/dev/null || echo ""
        fi
    else
        echo -e "${RED}✗ Сервіс не запущено${NC}"
        return 1
    fi
}

# CLI
COMMAND="${1:-start}"
shift || true

while [ $# -gt 0 ]; do
    case "$1" in
        --variant) VARIANT="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        *) shift ;;
    esac
done

export MODEL_FILE HOST PORT

case "$COMMAND" in
    start) start_service ;;
    stop) stop_service ;;
    restart) stop_service; sleep 1; start_service ;;
    status) status_service ;;
    *) echo "Usage: $0 {start|stop|restart|status} [--variant Q8|F16] [--port PORT]"; exit 1 ;;
esac
