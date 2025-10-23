#!/data/data/com.termux/files/usr/bin/bash
################################################################################
# AI Models Launcher - Інтерактивний запуск моделей
# Підтримує: Gemma 3N 2B + DeepSeek Coder 6.7B (чат) + Ukrainian MPNet Q8/F16 (ембеддинги)
# Платформа: Samsung Galaxy Tab S8 Pro (Snapdragon 8 Gen 1, Android 15)
################################################################################

set -e

# ══════════════════════════════════════════════════════════════════════════
# КОЛЬОРИ
# ══════════════════════════════════════════════════════════════════════════
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# ══════════════════════════════════════════════════════════════════════════
# ШЛЯХИ ДО МОДЕЛЕЙ
# ══════════════════════════════════════════════════════════════════════════
LLAMA_CLI="$HOME/llama.cpp/build/bin/llama-cli"
LLAMA_EMBEDDING="$HOME/llama.cpp/build/bin/llama-embedding"

# Чат моделі
GEMMA_2B="$HOME/models/gemma3n/google_gemma-3n-E2B-it-Q4_K_M.gguf"
DEEPSEEK_CODER="$HOME/models/deepseek-coder/deepseek-coder-6.7b-instruct.Q4_K_M.gguf"

# Ukrainian MPNet ембеддинги
MPNET_Q8="$HOME/models/embeddings/ukr-paraphrase-multilingual-mpnet-base-Q8_0.gguf"
MPNET_F16="$HOME/models/embeddings/ukr-paraphrase-multilingual-mpnet-base-F16.gguf"

# Логи
LOG_DIR="$HOME/models/logs"
mkdir -p "$LOG_DIR"

# ══════════════════════════════════════════════════════════════════════════
# ФУНКЦІЇ
# ══════════════════════════════════════════════════════════════════════════

print_header() {
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

check_file() {
    local file="$1"
    local name="$2"

    if [ ! -f "$file" ]; then
        print_error "$name не знайдено: $file"
        return 1
    fi
    return 0
}

get_file_size() {
    if [ -f "$1" ]; then
        du -h "$1" | cut -f1
    else
        echo "N/A"
    fi
}

check_ram() {
    local available=$(free -g 2>/dev/null | awk '/^Mem:/{print $7}' || echo "0")
    echo "$available"
}

get_temperature() {
    # Спроба отримати температуру CPU
    local temp=0
    for zone in /sys/class/thermal/thermal_zone*/temp; do
        if [ -f "$zone" ]; then
            local t=$(cat "$zone" 2>/dev/null || echo 0)
            temp=$((t / 1000))
            if [ $temp -gt 0 ]; then
                echo "$temp"
                return
            fi
        fi
    done
    echo "N/A"
}

get_running_models() {
    # Показує які AI моделі зараз працюють
    # ВАЖЛИВО: НЕ чіпаємо VPN процеси (smart_proxy, survey_automation)
    ps aux | grep -E 'llama-cli|python3.*embed|llama-server' | \
        grep -v grep | \
        grep -v smart_proxy | \
        grep -v survey_automation
}

count_running_models() {
    # Рахує кількість запущених моделей
    get_running_models | wc -l
}

kill_all_models() {
    # Вбиває всі запущені AI моделі
    local count=$(count_running_models)

    if [ "$count" -eq 0 ]; then
        return 0
    fi

    print_warning "Знайдено $count запущених процесів AI моделей"
    echo ""

    # Показати які процеси
    get_running_models | awk '{print "  PID " $2 ": " $11 " " $12 " " $13}' | head -5
    echo ""

    # Зупинка ембеддинг сервісу (якщо запущений)
    if [ -f ~/vpn/start_embedding_service.sh ]; then
        ~/vpn/start_embedding_service.sh stop 2>/dev/null || true
    fi

    # Вбити llama-cli процеси
    pkill -f llama-cli 2>/dev/null || true
    pkill -f llama-server 2>/dev/null || true

    # Вбити Python ембеддинг процеси (НЕ чіпаємо VPN!)
    ps aux | grep -E 'python3.*embed' | \
        grep -v grep | \
        grep -v smart_proxy | \
        grep -v survey_automation | \
        awk '{print $2}' | while read pid; do
        kill "$pid" 2>/dev/null || true
    done

    sleep 1

    # Перевірка
    local remaining=$(count_running_models)
    if [ "$remaining" -eq 0 ]; then
        print_success "Всі AI процеси зупинено"
    else
        print_warning "Залишилось $remaining процесів (примусове завершення...)"
        pkill -9 -f llama-cli 2>/dev/null || true
        pkill -9 -f llama-server 2>/dev/null || true
        # Примусове завершення Python ембеддингів (НЕ чіпаємо VPN!)
        ps aux | grep -E 'python3.*embed' | \
            grep -v grep | \
            grep -v smart_proxy | \
            grep -v survey_automation | \
            awk '{print $2}' | xargs -r kill -9 2>/dev/null || true
        sleep 1
        print_success "Примусово зупинено"
    fi

    echo ""
}

show_system_info() {
    clear
    print_header "📊 Інформація про систему"
    echo ""
    echo -e "${CYAN}Пристрій:${NC} $(getprop ro.product.model 2>/dev/null || echo 'Unknown')"
    echo -e "${CYAN}Android:${NC} $(getprop ro.build.version.release 2>/dev/null || echo 'N/A')"
    echo -e "${CYAN}CPU:${NC} $(uname -m)"
    echo -e "${CYAN}Доступна RAM:${NC} $(check_ram)GB"
    echo -e "${CYAN}Температура CPU:${NC} $(get_temperature)°C"
    echo ""
}

get_tailscale_ip() {
    # Спроба отримати Tailscale IP (100.x.x.x діапазон)
    local ts_ip=$(ifconfig 2>/dev/null | grep "inet 100\." | awk '{print $2}' | head -1)
    if [ -z "$ts_ip" ]; then
        ts_ip=$(ip addr show 2>/dev/null | grep "inet 100\." | awk '{print $2}' | cut -d/ -f1 | head -1)
    fi
    if [ -z "$ts_ip" ]; then
        ts_ip=$(tailscale ip 2>/dev/null)
    fi
    echo "$ts_ip"
}

show_models_status() {
    print_header "📦 Статус моделей"
    echo ""

    # Чат моделі
    echo -e "${YELLOW}🤖 Чат-моделі:${NC}"
    if check_file "$GEMMA_2B" "Gemma 3N 2B" 2>/dev/null; then
        print_success "Gemma 3N 2B E2B-it-Q4_K_M ($(get_file_size "$GEMMA_2B"))"
    else
        print_error "Gemma 3N 2B не знайдено: $GEMMA_2B"
    fi

    if check_file "$DEEPSEEK_CODER" "DeepSeek Coder" 2>/dev/null; then
        print_success "DeepSeek Coder 6.7B Q4_K_M ($(get_file_size "$DEEPSEEK_CODER"))"
    else
        print_error "DeepSeek Coder не знайдено: $DEEPSEEK_CODER"
    fi

    echo ""

    # MPNet ембеддинги
    echo -e "${YELLOW}🇺🇦 Ukrainian MPNet (Ембеддинги):${NC}"
    if check_file "$MPNET_Q8" "MPNet Q8" 2>/dev/null; then
        print_success "MPNet Q8_0 ($(get_file_size "$MPNET_Q8")) - швидкий"
    else
        print_error "MPNet Q8 - не встановлено"
    fi

    if check_file "$MPNET_F16" "MPNet F16" 2>/dev/null; then
        print_success "MPNet F16 ($(get_file_size "$MPNET_F16")) - точний"
    else
        print_error "MPNet F16 - не встановлено"
    fi

    echo ""

    # Запущені HTTP сервери
    echo -e "${YELLOW}🌐 Запущені HTTP сервери:${NC}"
    local has_servers=false

    # Перевірка Gemma 2B
    if curl -s http://127.0.0.1:8080/health &>/dev/null; then
        print_success "Gemma 2B Server :8080"
        has_servers=true
    fi

    # Перевірка DeepSeek Coder
    if curl -s http://127.0.0.1:8081/health &>/dev/null; then
        print_success "DeepSeek Coder Server :8081"
        has_servers=true
    fi

    # Перевірка Ембеддингів
    if curl -s http://127.0.0.1:8765/health &>/dev/null; then
        print_success "Ukrainian MPNet :8765"
        has_servers=true
    fi

    if [ "$has_servers" = false ]; then
        print_info "Жодного сервера не запущено"
    fi

    # Tailscale IP для віддаленого доступу
    local ts_ip=$(get_tailscale_ip)
    if [ -n "$ts_ip" ]; then
        echo ""
        echo -e "${CYAN}🔗 Tailscale:${NC} $ts_ip (віддалений доступ)"
        if [ "$has_servers" = true ]; then
            echo -e "   ${BLUE}Приклад:${NC} curl http://$ts_ip:8080/health"
        fi
    fi

    echo ""
}

# ══════════════════════════════════════════════════════════════════════════
# ФУНКЦІЇ ЗАПУСКУ МОДЕЛЕЙ
# ══════════════════════════════════════════════════════════════════════════

start_gemma_2b() {
    if ! check_file "$LLAMA_CLI" "llama-cli"; then
        print_error "Встанови llama.cpp спочатку"
        return 1
    fi

    if ! check_file "$GEMMA_2B" "Gemma 3N 2B"; then
        print_error "Модель не знайдена: $GEMMA_2B"
        return 1
    fi

    # Зупинити інші AI моделі перед запуском
    kill_all_models

    clear
    print_header "🚀 Запуск Gemma 3N 2B E2B-it-Q4_K_M (Швидкий чат)"
    echo ""
    print_info "Модель: $(basename $GEMMA_2B)"
    print_info "Розмір: $(get_file_size $GEMMA_2B)"
    print_info "Threads: 6 (CPU 0-5: A510 + A710)"
    print_info "Context: 2048 tokens"
    print_info "Швидкість: ~15-25 tokens/sec"
    echo ""
    print_warning "Натисни Ctrl+C для виходу"
    echo ""

    sleep 2

    taskset -c 0-5 "$LLAMA_CLI" \
        -m "$GEMMA_2B" \
        -t 6 \
        -c 2048 \
        -n -1 \
        --temp 0.7 \
        -ngl 0 \
        --color \
        -i \
        -p "Ти корисний AI асистент. Відповідай українською мовою."
}

start_deepseek_coder() {
    if ! check_file "$LLAMA_CLI" "llama-cli"; then
        print_error "Встанови llama.cpp спочатку"
        return 1
    fi

    if ! check_file "$DEEPSEEK_CODER" "DeepSeek Coder"; then
        print_error "Модель не знайдена: $DEEPSEEK_CODER"
        return 1
    fi

    # Перевірка RAM
    local available_ram=$(check_ram)
    if [ "$available_ram" != "N/A" ] && [ "$available_ram" -lt 5 ]; then
        print_warning "Доступно лише ${available_ram}GB RAM (потрібно 5GB+)"
        echo ""
        read -p "Продовжити? (y/n): " confirm
        if [ "$confirm" != "y" ]; then
            return 0
        fi
    fi

    # Зупинити інші AI моделі перед запуском
    kill_all_models

    clear
    print_header "🚀 Запуск DeepSeek Coder 6.7B Q4_K_M (Програмування)"
    echo ""
    print_warning "ВЕЛИКА МОДЕЛЬ! Потребує ~5GB RAM"
    print_info "Модель: $(basename $DEEPSEEK_CODER)"
    print_info "Розмір: $(get_file_size $DEEPSEEK_CODER)"
    print_info "Threads: 7 (CPU 0-6: всі крім X2)"
    print_info "Context: 4096 tokens"
    print_info "Швидкість: ~5-10 tokens/sec"
    print_info "Спеціалізація: Python, JavaScript, C++, Java"
    echo ""
    print_warning "Натисни Ctrl+C для виходу"
    echo ""

    sleep 3

    taskset -c 0-6 "$LLAMA_CLI" \
        -m "$DEEPSEEK_CODER" \
        -t 7 \
        -c 4096 \
        -n -1 \
        --temp 0.7 \
        -ngl 0 \
        --mlock \
        --color \
        -i \
        -p "You are an expert programming assistant. Help with code, explain concepts, and provide solutions."
}

start_mpnet_q8() {
    if ! check_file "$MPNET_Q8" "MPNet Q8"; then
        print_error "Модель не знайдена. Встанови: ./install_embeddings.sh"
        return 1
    fi

    # Зупинити інші AI моделі перед запуском
    kill_all_models

    clear
    print_header "🇺🇦 Ukrainian MPNet Q8_0 (Швидкі ембеддинги)"
    echo ""
    print_info "Модель: $(basename $MPNET_Q8)"
    print_info "Розмір: $(get_file_size $MPNET_Q8)"
    print_info "Threads: 6"
    print_info "Dimension: 768"
    echo ""

    # Запуск HTTP сервера для ембеддингів
    print_info "Запуск HTTP сервера на порту 8765..."
    echo ""

    cd ~/vpn
    if [ -f "start_embedding_service.sh" ]; then
        ./start_embedding_service.sh start --variant Q8
    else
        print_error "start_embedding_service.sh не знайдено"
    fi
}

start_mpnet_f16() {
    if ! check_file "$MPNET_F16" "MPNet F16"; then
        print_error "Модель не знайдена. Встанови: ./install_embeddings.sh"
        return 1
    fi

    # Зупинити інші AI моделі перед запуском
    kill_all_models

    clear
    print_header "🇺🇦 Ukrainian MPNet F16 (Точні ембеддинги)"
    echo ""
    print_info "Модель: $(basename $MPNET_F16)"
    print_info "Розмір: $(get_file_size $MPNET_F16)"
    print_info "Threads: 6"
    print_info "Dimension: 768"
    echo ""
    print_warning "Потребує ~600MB RAM"
    echo ""

    # Запуск HTTP сервера
    print_info "Запуск HTTP сервера на порту 8765..."
    echo ""

    cd ~/vpn
    if [ -f "start_embedding_service.sh" ]; then
        ./start_embedding_service.sh start --variant F16
    else
        print_error "start_embedding_service.sh не знайдено"
    fi
}

test_embeddings() {
    clear
    print_header "🧪 Тестування Ukrainian MPNet"
    echo ""

    cd ~/vpn
    if [ -f "test_embedding_uk.sh" ]; then
        ./test_embedding_uk.sh
    else
        print_error "test_embedding_uk.sh не знайдено"
    fi
}

# ══════════════════════════════════════════════════════════════════════════
# GEMMA HTTP SERVER ФУНКЦІЇ
# ══════════════════════════════════════════════════════════════════════════

start_gemma_2b_server() {
    clear
    print_header "🚀 Запуск Gemma 2B HTTP Server"
    echo ""

    # Зупинити інші AI моделі перед запуском
    kill_all_models

    cd ~/vpn
    if [ -f "start_gemma_service.sh" ]; then
        ./start_gemma_service.sh start --variant 2B --port 8080
        echo ""
        print_success "Gemma 2B Server запущено!"
        print_info "API: http://127.0.0.1:8080"
        echo ""
        print_info "Приклад curl:"
        echo '  curl http://127.0.0.1:8080/completion -H "Content-Type: application/json" -d '"'"'{"prompt":"Привіт!","n_predict":50}'"'"''
    else
        print_error "start_gemma_service.sh не знайдено"
    fi

    echo ""
    read -p "Натисни Enter для продовження..."
}

start_deepseek_coder_server() {
    clear
    print_header "🚀 Запуск DeepSeek Coder HTTP Server"
    echo ""
    print_warning "ВЕЛИКА МОДЕЛЬ! Потребує ~5GB RAM"
    echo ""

    if ! check_file "$DEEPSEEK_CODER" "DeepSeek Coder"; then
        print_error "Модель не знайдена: $DEEPSEEK_CODER"
        echo ""
        read -p "Натисни Enter для продовження..."
        return 1
    fi

    # Зупинити інші AI моделі перед запуском
    kill_all_models

    print_info "Запуск llama-server..."
    nohup ~/llama.cpp/build/bin/llama-server \
        -m "$DEEPSEEK_CODER" \
        --host 127.0.0.1 \
        --port 8081 \
        -t 7 \
        -c 4096 \
        --temp 0.7 \
        -ngl 0 \
        --log-disable > ~/models/logs/deepseek-coder.log 2>&1 &

    sleep 3

    if curl -s http://127.0.0.1:8081/health &>/dev/null; then
        print_success "DeepSeek Coder Server запущено!"
        print_info "API: http://127.0.0.1:8081"
        echo ""
        print_info "Приклад curl:"
        echo '  curl http://127.0.0.1:8081/completion -H "Content-Type: application/json" -d '"'"'{"prompt":"Write a Python function","n_predict":100}'"'"''
    else
        print_error "Не вдалося запустити сервер"
    fi

    echo ""
    read -p "Натисни Enter для продовження..."
}

server_status() {
    clear
    print_header "📊 Статус HTTP Серверів"
    echo ""

    echo -e "${CYAN}Перевірка Gemma 2B (порт 8080):${NC}"
    if curl -s http://127.0.0.1:8080/health &>/dev/null; then
        print_success "Gemma 2B Server працює"
        curl -s http://127.0.0.1:8080/health 2>/dev/null | head -5
    else
        print_error "Gemma 2B Server не запущено"
    fi

    echo ""
    echo -e "${CYAN}Перевірка DeepSeek Coder (порт 8081):${NC}"
    if curl -s http://127.0.0.1:8081/health &>/dev/null; then
        print_success "DeepSeek Coder Server працює"
        curl -s http://127.0.0.1:8081/health 2>/dev/null | head -5
    else
        print_error "DeepSeek Coder Server не запущено"
    fi

    echo ""
    echo -e "${CYAN}Перевірка Ukrainian MPNet (порт 8765):${NC}"
    if curl -s http://127.0.0.1:8765/health &>/dev/null; then
        print_success "Ukrainian MPNet працює"
        curl -s http://127.0.0.1:8765/health 2>/dev/null | head -5
    else
        print_error "Ukrainian MPNet не запущено"
    fi

    echo ""
    read -p "Натисни Enter для продовження..."
}

test_api() {
    clear
    print_header "🧪 Тест HTTP API"
    echo ""

    # Перевірити який сервер працює
    local port=""
    local model_name=""
    local test_prompt=""

    if curl -s http://127.0.0.1:8080/health &>/dev/null; then
        port="8080"
        model_name="Gemma 2B"
        test_prompt="Привіт! Розкажи коротко про себе українською."
        print_success "Використовую Gemma 2B на порту 8080"
    elif curl -s http://127.0.0.1:8081/health &>/dev/null; then
        port="8081"
        model_name="DeepSeek Coder"
        test_prompt="Write a Python function to calculate fibonacci numbers:"
        print_success "Використовую DeepSeek Coder на порту 8081"
    else
        print_error "Жоден сервер не запущено!"
        echo ""
        print_info "Спочатку запусти сервер (опція 11 або 12)"
        echo ""
        read -p "Натисни Enter для продовження..."
        return 1
    fi

    echo ""
    echo -e "${YELLOW}Відправляю запит до $model_name...${NC}"
    echo ""

    curl -s http://127.0.0.1:$port/completion \
        -H "Content-Type: application/json" \
        -d "{\"prompt\":\"$test_prompt\",\"n_predict\":100,\"temperature\":0.7}" | \
        python3 -c "import sys,json; d=json.load(sys.stdin); print('Відповідь:', d.get('content', 'N/A'))" 2>/dev/null || \
        print_error "Помилка при запиті"

    echo ""
    read -p "Натисни Enter для продовження..."
}

# ══════════════════════════════════════════════════════════════════════════
# GEMMA REMOTE SERVER ФУНКЦІЇ (TAILSCALE)
# ══════════════════════════════════════════════════════════════════════════

start_gemma_2b_remote() {
    clear
    print_header "🚀 Запуск Gemma 2B HTTP Server (Tailscale)"
    echo ""

    # Перевірка Tailscale
    local ts_ip=$(get_tailscale_ip)
    if [ -z "$ts_ip" ]; then
        print_warning "Tailscale IP не знайдено!"
        echo ""
        print_info "Сервер запуститься на 0.0.0.0:8080"
        print_info "Доступ буде через всі мережеві інтерфейси"
        echo ""
    else
        print_success "Tailscale IP: $ts_ip"
        echo ""
    fi

    # Зупинити інші AI моделі перед запуском
    kill_all_models

    cd ~/vpn
    if [ -f "start_gemma_service.sh" ]; then
        ./start_gemma_service.sh start --variant 2B --port 8080 --host 0.0.0.0
        echo ""
        print_success "Gemma 2B Server запущено (0.0.0.0:8080)!"
        echo ""
        print_info "Локальний доступ:    http://127.0.0.1:8080"
        if [ -n "$ts_ip" ]; then
            print_info "Tailscale доступ:    http://$ts_ip:8080"
        fi
    else
        print_error "start_gemma_service.sh не знайдено"
    fi

    echo ""
    read -p "Натисни Enter для продовження..."
}

start_deepseek_coder_remote() {
    clear
    print_header "🚀 Запуск DeepSeek Coder HTTP Server (Tailscale)"
    echo ""
    print_warning "ВЕЛИКА МОДЕЛЬ! Потребує ~5GB RAM"
    echo ""

    if ! check_file "$DEEPSEEK_CODER" "DeepSeek Coder"; then
        print_error "Модель не знайдена: $DEEPSEEK_CODER"
        echo ""
        read -p "Натисни Enter для продовження..."
        return 1
    fi

    # Перевірка Tailscale
    local ts_ip=$(get_tailscale_ip)
    if [ -z "$ts_ip" ]; then
        print_warning "Tailscale IP не знайдено!"
        echo ""
        print_info "Сервер запуститься на 0.0.0.0:8081"
        print_info "Доступ буде через всі мережеві інтерфейси"
        echo ""
    else
        print_success "Tailscale IP: $ts_ip"
        echo ""
    fi

    # Зупинити інші AI моделі перед запуском
    kill_all_models

    print_info "Запуск llama-server..."
    nohup ~/llama.cpp/build/bin/llama-server \
        -m "$DEEPSEEK_CODER" \
        --host 0.0.0.0 \
        --port 8081 \
        -t 7 \
        -c 4096 \
        --temp 0.7 \
        -ngl 0 \
        --log-disable > ~/models/logs/deepseek-coder-remote.log 2>&1 &

    sleep 3

    if curl -s http://127.0.0.1:8081/health &>/dev/null; then
        echo ""
        print_success "DeepSeek Coder Server запущено (0.0.0.0:8081)!"
        echo ""
        print_info "Локальний доступ:    http://127.0.0.1:8081"
        if [ -n "$ts_ip" ]; then
            print_info "Tailscale доступ:    http://$ts_ip:8081"
        fi
    else
        print_error "Не вдалося запустити сервер"
    fi

    echo ""
    read -p "Натисни Enter для продовження..."
}

# ══════════════════════════════════════════════════════════════════════════
# ГОЛОВНЕ МЕНЮ
# ══════════════════════════════════════════════════════════════════════════

show_menu() {
    clear
    show_system_info
    show_models_status

    print_header "🎯 AI Models Launcher - Головне меню"
    echo ""
    echo -e "${GREEN}Чат моделі - Інтерактивний режим:${NC}"
    echo "  1) Gemma 2B            - швидкий чат (2.6GB, ~20 tok/s)"
    echo "  2) DeepSeek Coder      - програмування (3.9GB, ~5 tok/s) ⚠️"
    echo ""
    echo -e "${GREEN}HTTP Server (локальний 127.0.0.1):${NC}"
    echo "  11) Gemma 2B           - HTTP API :8080"
    echo "  12) DeepSeek Coder     - HTTP API :8081"
    echo ""
    echo -e "${GREEN}HTTP Server (Tailscale віддалений):${NC}"
    echo "  21) Gemma 2B           - HTTP API :8080 (0.0.0.0)"
    echo "  22) DeepSeek Coder     - HTTP API :8081 (0.0.0.0)"
    echo ""
    echo -e "${GREEN}Керування серверами:${NC}"
    echo "  13) Статус серверів    - перевірити статус"
    echo "  14) Тест API           - швидкий тест HTTP запиту"
    echo ""
    echo -e "${GREEN}Ukrainian MPNet (Ембеддинги):${NC}"
    echo "  3) MPNet Q8_0          - швидкі ембеддинги HTTP :8765 (290MB)"
    echo "  4) MPNet F16           - точні ембеддинги HTTP :8765 (538MB)"
    echo ""
    echo -e "${GREEN}Інше:${NC}"
    echo "  5) Тестування українських ембеддингів"
    echo "  6) Моніторинг температури CPU"
    echo "  7) Перегляд логів"
    echo "  8) 🛑 Зупинити всі AI моделі"
    echo ""
    echo "  0) Вихід"
    echo ""
    echo -e -n "${CYAN}Вибери опцію [0-22]: ${NC}"
}

monitor_thermal() {
    clear
    print_header "🌡️  Моніторинг температури CPU"
    echo ""
    print_info "Оновлення кожні 2 секунди. Натисни Ctrl+C для виходу"
    echo ""

    while true; do
        local temp=$(get_temperature)
        local ram=$(check_ram)

        echo -ne "\r${CYAN}CPU Temp:${NC} ${temp}°C  |  ${CYAN}RAM:${NC} ${ram}GB free  "

        if [ "$temp" != "N/A" ] && [ "$temp" -gt 65 ]; then
            echo -ne "${RED}⚠️ ПЕРЕГРІВ!${NC}     "
        else
            echo -ne "${GREEN}✓ OK${NC}        "
        fi

        sleep 2
    done
}

view_logs() {
    clear
    print_header "📋 Логи"
    echo ""

    if [ -d "$LOG_DIR" ]; then
        ls -lh "$LOG_DIR"
        echo ""
        echo "Виберіть лог для перегляду (або Enter для повернення):"
        read -p "Файл: " logfile

        if [ -n "$logfile" ] && [ -f "$LOG_DIR/$logfile" ]; then
            less "$LOG_DIR/$logfile"
        fi
    else
        print_info "Логи порожні"
    fi

    echo ""
    read -p "Натисни Enter для продовження..."
}

# ══════════════════════════════════════════════════════════════════════════
# ГОЛОВНИЙ ЦИКЛ
# ══════════════════════════════════════════════════════════════════════════

main() {
    while true; do
        show_menu
        read choice

        case $choice in
            1)
                start_gemma_2b
                ;;
            2)
                start_deepseek_coder
                ;;
            3)
                start_mpnet_q8
                ;;
            4)
                start_mpnet_f16
                ;;
            5)
                test_embeddings
                read -p "Натисни Enter для продовження..."
                ;;
            6)
                monitor_thermal
                ;;
            7)
                view_logs
                ;;
            8)
                clear
                print_header "🛑 Зупинка всіх AI моделей"
                echo ""
                kill_all_models
                echo ""
                read -p "Натисни Enter для продовження..."
                ;;
            11)
                start_gemma_2b_server
                ;;
            12)
                start_deepseek_coder_server
                ;;
            13)
                server_status
                ;;
            14)
                test_api
                ;;
            21)
                start_gemma_2b_remote
                ;;
            22)
                start_deepseek_coder_remote
                ;;
            0)
                clear
                print_success "До побачення!"
                exit 0
                ;;
            *)
                print_error "Невірний вибір"
                sleep 2
                ;;
        esac
    done
}

# Запуск
main
