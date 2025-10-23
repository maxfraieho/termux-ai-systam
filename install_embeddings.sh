#!/usr/bin/env bash
set -euo pipefail

# ════════════════════════════════════════════════════════════════════════════
# Скрипт завантаження і встановлення Ukrainian MPNet GGUF моделей
# Підтримує: F16 (точний, ~563MB) і Q8_0 (легкий, ~303MB)
# Платформа: ARM64 Android Termux/Droidian
# ════════════════════════════════════════════════════════════════════════════

# ──────────────────────────────────────────────────────────────────────────
# КОНФІГУРАЦІЯ
# ──────────────────────────────────────────────────────────────────────────
HF_TOKEN="${HF_TOKEN:-}"
REPO="podarok/ukr-paraphrase-multilingual-mpnet-base"
Q8_FILE="ukr-paraphrase-multilingual-mpnet-base-Q8_0.gguf"
F16_FILE="ukr-paraphrase-multilingual-mpnet-base-F16.gguf"

# Шляхи (спробуємо у порядку пріоритету)
TARGET_DIRS=(
  "$HOME/models/embeddings"
  "$HOME/storage/shared/models/embeddings"
  "/storage/emulated/0/models/embeddings"
)

TMP_DIR="${TMPDIR:-$HOME/tmp}/ukr_mpnet_install_$$"
REPORT_DIR="$HOME/models/ukr-mpnet"
REPORT="$REPORT_DIR/install_report.txt"
OPT_DIR="/opt/gguf/embeddings"
INDEX_FILE="$HOME/models/models_index.json"

# HuggingFace URLs
Q8_URL="https://huggingface.co/${REPO}/resolve/main/${Q8_FILE}"
F16_URL="https://huggingface.co/${REPO}/resolve/main/${F16_FILE}"

# Очікувані контрольні суми (залиш порожніми, якщо невідомі)
Q8_SHA256="${Q8_SHA256:-}"
F16_SHA256="${F16_SHA256:-}"

# ──────────────────────────────────────────────────────────────────────────
# ДОПОМІЖНІ ФУНКЦІЇ
# ──────────────────────────────────────────────────────────────────────────

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$REPORT"
}

check_prerequisites() {
  log "Перевірка системних вимог..."

  # Перевірка архітектури
  if [[ "$(uname -m)" != "aarch64" && "$(uname -m)" != "arm64" ]]; then
    log "УВАГА: Система не ARM64. Модель може працювати неоптимально."
  fi

  # Перевірка вільного місця (потрібно ~1GB)
  local free_space=$(df -h "$HOME" | awk 'NR==2 {print $4}' | sed 's/[^0-9.]//g')
  log "Вільне місце: ${free_space}G"
  # Пропускаємо перевірку якщо не вдалося розпарсити
  if [[ -n "$free_space" && $(echo "$free_space < 1" | bc -l 2>/dev/null || echo 0) -eq 1 ]]; then
    log "ПОПЕРЕДЖЕННЯ: Може бути недостатньо місця (потрібно ~1GB)"
  fi

  # Перевірка Python
  if ! command -v python3 >/dev/null 2>&1; then
    log "УВАГА: Python3 не знайдено. Встанови: pkg install python"
  fi

  # Перевірка curl/wget
  if ! command -v curl >/dev/null 2>&1 && ! command -v wget >/dev/null 2>&1; then
    log "ПОМИЛКА: Потрібен curl або wget. Встанови: pkg install curl"
    exit 1
  fi
}

download_with_python() {
  local repo="$1" filename="$2" output="$3"

  python3 - <<PYTHON
import sys
import os

try:
    from huggingface_hub import hf_hub_download
    from tqdm import tqdm

    token = os.environ.get("HF_TOKEN")

    print(f"Завантаження через huggingface_hub: ${filename}")

    downloaded_path = hf_hub_download(
        repo_id="${repo}",
        filename="${filename}",
        token=token,
        cache_dir="${TMP_DIR}/.cache"
    )

    # Копіюємо до цільового місця
    import shutil
    shutil.copy2(downloaded_path, "${output}")

    print(f"SUCCESS:{downloaded_path}")
    sys.exit(0)

except ImportError:
    print("ERROR:huggingface_hub не встановлено. Використовуй: pip install huggingface_hub", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"ERROR:{str(e)}", file=sys.stderr)
    sys.exit(1)
PYTHON
}

download_with_curl() {
  local url="$1" output="$2"

  log "Завантаження через curl: $url"

  local curl_opts="-L --progress-bar --fail --retry 3 --retry-delay 5"

  if [[ -n "$HF_TOKEN" ]]; then
    curl_opts="$curl_opts -H 'Authorization: Bearer $HF_TOKEN'"
  fi

  if curl $curl_opts -o "$output" "$url"; then
    return 0
  else
    return 1
  fi
}

download_file() {
  local url="$1" filename="$2" output="$3"

  log "Початок завантаження: $filename"

  # Спроба 1: Python з huggingface_hub
  if command -v python3 >/dev/null 2>&1; then
    if download_with_python "$REPO" "$filename" "$output" 2>&1 | tee -a "$REPORT"; then
      if [[ -f "$output" ]]; then
        log "✓ Успішно завантажено через Python"
        return 0
      fi
    fi
  fi

  # Спроба 2: curl
  if command -v curl >/dev/null 2>&1; then
    if download_with_curl "$url" "$output"; then
      log "✓ Успішно завантажено через curl"
      return 0
    fi
  fi

  # Спроба 3: wget
  if command -v wget >/dev/null 2>&1; then
    log "Завантаження через wget: $url"
    if wget -c -O "$output" "$url"; then
      log "✓ Успішно завантажено через wget"
      return 0
    fi
  fi

  log "✗ ПОМИЛКА: Не вдалося завантажити $filename"
  return 1
}

verify_checksum() {
  local file="$1" expected="$2"

  if [[ -z "$expected" ]]; then
    log "Контрольна сума не задана, пропускаю перевірку"
    return 0
  fi

  log "Перевірка SHA256 для $(basename "$file")..."

  local actual=$(sha256sum "$file" | awk '{print $1}')

  log "  Очікується: $expected"
  log "  Фактично:   $actual"

  if [[ "$actual" == "$expected" ]]; then
    log "✓ Контрольна сума збігається"
    return 0
  else
    log "✗ ПОМИЛКА: Контрольна сума НЕ збігається!"
    return 1
  fi
}

create_symlinks() {
  local dest_dir="$1"

  log "Створення симлінків у $OPT_DIR..."

  # Спроба створити каталог (з sudo якщо доступний)
  if command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; then
    sudo mkdir -p "$OPT_DIR" 2>/dev/null || mkdir -p "$HOME/.local/opt/gguf/embeddings"
    OPT_DIR="${OPT_DIR:-$HOME/.local/opt/gguf/embeddings}"

    sudo ln -sf "$dest_dir/$Q8_FILE" "$OPT_DIR/lang-uk-mpnet-Q8.gguf"
    sudo ln -sf "$dest_dir/$F16_FILE" "$OPT_DIR/lang-uk-mpnet-F16.gguf"
  else
    # Fallback: домашня тека
    OPT_DIR="$HOME/.local/opt/gguf/embeddings"
    mkdir -p "$OPT_DIR"
    ln -sf "$dest_dir/$Q8_FILE" "$OPT_DIR/lang-uk-mpnet-Q8.gguf"
    ln -sf "$dest_dir/$F16_FILE" "$OPT_DIR/lang-uk-mpnet-F16.gguf"
  fi

  log "✓ Симлінки створено:"
  log "  Q8:  $OPT_DIR/lang-uk-mpnet-Q8.gguf -> $dest_dir/$Q8_FILE"
  log "  F16: $OPT_DIR/lang-uk-mpnet-F16.gguf -> $dest_dir/$F16_FILE"
}

update_index() {
  local dest_dir="$1"

  log "Оновлення моделевого індексу: $INDEX_FILE"

  mkdir -p "$(dirname "$INDEX_FILE")"

  if [[ ! -f "$INDEX_FILE" ]]; then
    echo '[]' > "$INDEX_FILE"
  fi

  python3 - <<PYTHON
import json
import os

index_file = "${INDEX_FILE}"

with open(index_file, "r") as f:
    try:
        data = json.load(f)
    except:
        data = []

# Видалити старий запис якщо є
data = [e for e in data if e.get("id") != "lang-uk/ukr-paraphrase-multilingual-mpnet-base"]

# Додати новий запис
entry = {
    "id": "lang-uk/ukr-paraphrase-multilingual-mpnet-base",
    "name": "Ukrainian Paraphrase Multilingual MPNet",
    "description": "Українська мультилінгвальна модель для генерації ембеддингів",
    "architecture": "MPNet",
    "context_length": 512,
    "variants": [
        {
            "tag": "F16",
            "local_path": "${dest_dir}/${F16_FILE}",
            "symlink": "${OPT_DIR}/lang-uk-mpnet-F16.gguf",
            "format": "GGUF",
            "quant": "F16",
            "dim": 768,
            "size_mb": 563,
            "source": "huggingface://${REPO}",
            "recommended_for": "Максимальна точність, багато RAM"
        },
        {
            "tag": "Q8_0",
            "local_path": "${dest_dir}/${Q8_FILE}",
            "symlink": "${OPT_DIR}/lang-uk-mpnet-Q8.gguf",
            "format": "GGUF",
            "quant": "Q8_0",
            "dim": 768,
            "size_mb": 303,
            "source": "huggingface://${REPO}",
            "recommended_for": "Баланс швидкості та точності (за замовчуванням)"
        }
    ],
    "install_date": "$(date -Iseconds)"
}

data.append(entry)

with open(index_file, "w") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print(f"✓ Індекс оновлено: {index_file}")
PYTHON

  log "✓ models_index.json оновлено"
}

# ──────────────────────────────────────────────────────────────────────────
# ГОЛОВНА ЛОГІКА
# ──────────────────────────────────────────────────────────────────────────

main() {
  # Підготовка
  mkdir -p "$REPORT_DIR" "$TMP_DIR"

  echo "════════════════════════════════════════════════════════════════════════════" > "$REPORT"
  log "Інсталяція Ukrainian MPNet GGUF моделей"
  log "Репозиторій: $REPO"
  log "Пристрій: $(uname -m) ($(getprop ro.product.model 2>/dev/null || echo 'Unknown'))"
  log "Android: $(getprop ro.build.version.release 2>/dev/null || echo 'N/A')"
  echo "════════════════════════════════════════════════════════════════════════════" >> "$REPORT"

  check_prerequisites

  # Визначення цільової теки
  DEST_DIR=""
  for dir in "${TARGET_DIRS[@]}"; do
    if mkdir -p "$dir" 2>/dev/null; then
      DEST_DIR="$dir"
      log "Обрано цільовий каталог: $DEST_DIR"
      break
    fi
  done

  if [[ -z "$DEST_DIR" ]]; then
    log "ПОМИЛКА: Не вдалося створити жодну з цільових тек!"
    exit 1
  fi

  # ──────────────────────────────────────────────────────────────────────
  # ЗАВАНТАЖЕННЯ Q8_0
  # ──────────────────────────────────────────────────────────────────────
  log ""
  log "┌─────────────────────────────────────────────────────────────────────┐"
  log "│ Завантаження Q8_0 (~303 MB)                                         │"
  log "└─────────────────────────────────────────────────────────────────────┘"

  if ! download_file "$Q8_URL" "$Q8_FILE" "$TMP_DIR/$Q8_FILE"; then
    log "КРИТИЧНА ПОМИЛКА: Не вдалося завантажити Q8_0"
    exit 1
  fi

  Q8_SIZE=$(du -h "$TMP_DIR/$Q8_FILE" | cut -f1)
  log "Розмір файлу: $Q8_SIZE"

  Q8_ACTUAL_SHA=$(sha256sum "$TMP_DIR/$Q8_FILE" | awk '{print $1}')
  log "SHA256: $Q8_ACTUAL_SHA"

  if [[ -n "$Q8_SHA256" ]]; then
    verify_checksum "$TMP_DIR/$Q8_FILE" "$Q8_SHA256" || exit 1
  fi

  # ──────────────────────────────────────────────────────────────────────
  # ЗАВАНТАЖЕННЯ F16
  # ──────────────────────────────────────────────────────────────────────
  log ""
  log "┌─────────────────────────────────────────────────────────────────────┐"
  log "│ Завантаження F16 (~563 MB)                                          │"
  log "└─────────────────────────────────────────────────────────────────────┘"

  if ! download_file "$F16_URL" "$F16_FILE" "$TMP_DIR/$F16_FILE"; then
    log "КРИТИЧНА ПОМИЛКА: Не вдалося завантажити F16"
    exit 1
  fi

  F16_SIZE=$(du -h "$TMP_DIR/$F16_FILE" | cut -f1)
  log "Розмір файлу: $F16_SIZE"

  F16_ACTUAL_SHA=$(sha256sum "$TMP_DIR/$F16_FILE" | awk '{print $1}')
  log "SHA256: $F16_ACTUAL_SHA"

  if [[ -n "$F16_SHA256" ]]; then
    verify_checksum "$TMP_DIR/$F16_FILE" "$F16_SHA256" || exit 1
  fi

  # ──────────────────────────────────────────────────────────────────────
  # ПЕРЕМІЩЕННЯ ФАЙЛІВ
  # ──────────────────────────────────────────────────────────────────────
  log ""
  log "Переміщення файлів до $DEST_DIR..."

  mv -v "$TMP_DIR/$Q8_FILE" "$DEST_DIR/" 2>&1 | tee -a "$REPORT"
  mv -v "$TMP_DIR/$F16_FILE" "$DEST_DIR/" 2>&1 | tee -a "$REPORT"

  log "✓ Файли переміщено"

  # ──────────────────────────────────────────────────────────────────────
  # СТВОРЕННЯ СИМЛІНКІВ
  # ──────────────────────────────────────────────────────────────────────
  create_symlinks "$DEST_DIR"

  # ──────────────────────────────────────────────────────────────────────
  # ОНОВЛЕННЯ ІНДЕКСУ
  # ──────────────────────────────────────────────────────────────────────
  update_index "$DEST_DIR"

  # ──────────────────────────────────────────────────────────────────────
  # ПІДСУМОК
  # ──────────────────────────────────────────────────────────────────────
  log ""
  log "════════════════════════════════════════════════════════════════════════════"
  log "✓ ІНСТАЛЯЦІЯ ЗАВЕРШЕНА УСПІШНО"
  log "════════════════════════════════════════════════════════════════════════════"
  log ""
  log "Встановлено моделі:"
  log "  • Q8_0: $DEST_DIR/$Q8_FILE ($Q8_SIZE)"
  log "  • F16:  $DEST_DIR/$F16_FILE ($F16_SIZE)"
  log ""
  log "Контрольні суми SHA256:"
  log "  Q8_0: $Q8_ACTUAL_SHA"
  log "  F16:  $F16_ACTUAL_SHA"
  log ""
  log "Симлінки:"
  log "  $OPT_DIR/lang-uk-mpnet-Q8.gguf"
  log "  $OPT_DIR/lang-uk-mpnet-F16.gguf"
  log ""
  log "Індекс моделей: $INDEX_FILE"
  log "Повний звіт: $REPORT"
  log ""
  log "Наступні кроки:"
  log "  1. Запуск сервісу: ./start_embedding_service.sh"
  log "  2. Тестування: ./test_embedding_uk.sh"
  log ""

  # Очищення
  rm -rf "$TMP_DIR"
}

# Запуск
trap 'log "Перервано користувачем"; rm -rf "$TMP_DIR"; exit 130' INT TERM
main "$@"
