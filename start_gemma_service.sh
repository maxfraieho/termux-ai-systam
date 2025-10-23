#!/data/data/com.termux/files/usr/bin/bash
################################################################################
# Gemma Chat Service - HTTP сервер для чат-моделей Gemma
# Працює в фоновому режимі, не залежить від терміналу
################################################################################

set -euo pipefail

# Конфігурація
LLAMA_SERVER="$HOME/llama.cpp/build/bin/llama-server"
MODEL_DIR="$HOME/models/gemma3n"
VARIANT="${VARIANT:-2B}"  # 2B або 4B
PORT="${PORT:-8080}"
HOST="${HOST:-127.0.0.1}"  # 127.0.0.1 = локально, 0.0.0.0 = віддалений доступ (Tailscale)
THREADS=6
CTX_SIZE=2048

PID_FILE="$HOME/models/gemma3n/service.pid"
LOG_FILE="$HOME/models/gemma3n/service.log"

# Кольори
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
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

    log "Зупинка Gemma сервісу (PID: $pid)..."

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
        log "Порт: http://$HOST:$PORT"
        exit 0
    fi

    # Визначення моделі
    case "$VARIANT" in
        2B|2b)
            MODEL_FILE="$MODEL_DIR/google_gemma-3n-E2B-it-Q4_K_M.gguf"
            THREADS=6
            ;;
        4B|4b)
            MODEL_FILE="$MODEL_DIR/gemma-3n-e4b-q4_k_m.gguf"
            THREADS=7
            ;;
        *)
            log "${RED}✗ Невідомий варіант '$VARIANT' (використай 2B або 4B)${NC}"
            exit 1
            ;;
    esac

    if [ ! -f "$MODEL_FILE" ]; then
        log "${RED}✗ Модель не знайдено: $MODEL_FILE${NC}"
        exit 1
    fi

    if [ ! -f "$LLAMA_SERVER" ]; then
        log "${RED}✗ llama-server не знайдено: $LLAMA_SERVER${NC}"
        exit 1
    fi

    mkdir -p "$(dirname "$LOG_FILE")"

    log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    log "🤖 Gemma $VARIANT Chat Service"
    log "Модель: $(basename $MODEL_FILE)"
    log "Bind: $HOST:$PORT"
    if [ "$HOST" = "0.0.0.0" ]; then
        log "Режим: Віддалений доступ (Tailscale)"
        # Отримати Tailscale IP якщо є (100.x.x.x діапазон)
        local ts_ip=$(ifconfig 2>/dev/null | grep "inet 100\." | awk '{print $2}' | head -1)
        if [ -n "$ts_ip" ]; then
            log "Tailscale: http://$ts_ip:$PORT"
        fi
    else
        log "Режим: Локальний доступ"
    fi
    log "Threads: $THREADS"
    log "Context: $CTX_SIZE tokens"
    log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # Запуск llama-server в фоновому режимі
    nohup taskset -c 0-$((THREADS-1)) "$LLAMA_SERVER" \
        -m "$MODEL_FILE" \
        --host "$HOST" \
        --port "$PORT" \
        -t "$THREADS" \
        -c "$CTX_SIZE" \
        --temp 0.7 \
        -ngl 0 \
        --log-disable \
        >> "$LOG_FILE" 2>&1 &

    local pid=$!
    echo "$pid" > "$PID_FILE"

    sleep 3

    if ! kill -0 "$pid" 2>/dev/null; then
        log "${RED}✗ Помилка запуску${NC}"
        cat "$LOG_FILE" | tail -20
        rm -f "$PID_FILE"
        exit 1
    fi

    log "${GREEN}✓ Сервіс запущено (PID: $pid)${NC}"
    log ""
    log "${CYAN}📡 API Endpoints:${NC}"

    if [ "$HOST" = "0.0.0.0" ]; then
        log "  Local:      http://127.0.0.1:$PORT/completion"
        local ts_ip=$(ifconfig 2>/dev/null | grep "inet 100\." | awk '{print $2}' | head -1)
        if [ -n "$ts_ip" ]; then
            log "  Tailscale:  http://$ts_ip:$PORT/completion"
        fi
    else
        log "  Completion: http://$HOST:$PORT/completion"
        log "  Chat:       http://$HOST:$PORT/v1/chat/completions"
        log "  Health:     http://$HOST:$PORT/health"
    fi

    log ""
    log "${CYAN}📝 Приклад curl запиту:${NC}"
    log "  curl http://127.0.0.1:$PORT/completion -H 'Content-Type: application/json' -d '{\"prompt\":\"Привіт! Як справи?\",\"n_predict\":100}'"
}

status_service() {
    if is_running; then
        echo -e "${GREEN}✓ Gemma $VARIANT працює (PID: $(get_service_pid))${NC}"
        echo -e "  API: http://$HOST:$PORT"

        # Спроба перевірити health
        if command -v curl >/dev/null 2>&1; then
            echo -e "\n${CYAN}Health check:${NC}"
            curl -s "http://$HOST:$PORT/health" 2>/dev/null || echo "  (сервер ще завантажується...)"
        fi
        return 0
    else
        echo -e "${RED}✗ Сервіс не запущено${NC}"
        return 1
    fi
}

test_chat() {
    if ! is_running; then
        echo -e "${RED}✗ Сервіс не запущено. Спочатку запусти: $0 start${NC}"
        exit 1
    fi

    echo -e "${CYAN}🧪 Тестування Gemma Chat API...${NC}\n"

    local prompt="Привіт! Розкажи коротко про себе українською мовою."

    echo -e "${YELLOW}Prompt:${NC} $prompt\n"

    curl -s "http://$HOST:$PORT/completion" \
        -H "Content-Type: application/json" \
        -d "{\"prompt\":\"$prompt\",\"n_predict\":150,\"temperature\":0.7}" | \
        python3 -c "import sys,json; print(json.load(sys.stdin)['content'])" 2>/dev/null || \
        echo -e "${RED}Помилка при запиті${NC}"
}

# CLI
COMMAND="${1:-start}"
shift || true

while [ $# -gt 0 ]; do
    case "$1" in
        --variant) VARIANT="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --host) HOST="$2"; shift 2 ;;
        *) shift ;;
    esac
done

case "$COMMAND" in
    start) start_service ;;
    stop) stop_service ;;
    restart) stop_service; sleep 1; start_service ;;
    status) status_service ;;
    test) test_chat ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|test} [OPTIONS]"
        echo ""
        echo "OPTIONS:"
        echo "  --variant 2B|4B    Модель (за замовчуванням: 2B)"
        echo "  --port PORT        HTTP порт (за замовчуванням: 8080)"
        echo "  --host HOST        Bind адреса (за замовчуванням: 127.0.0.1)"
        echo ""
        echo "Приклади:"
        echo "  # Локальний доступ:"
        echo "  $0 start --variant 2B --port 8080"
        echo ""
        echo "  # Віддалений доступ через Tailscale:"
        echo "  $0 start --variant 2B --port 8080 --host 0.0.0.0"
        echo ""
        echo "  # Інші команди:"
        echo "  $0 status"
        echo "  $0 test"
        echo "  $0 stop"
        exit 1
        ;;
esac
