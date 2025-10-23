# 🌐 VPN & AI Models - Samsung Galaxy Tab S8 Pro

Інтегрована система VPN та локальних AI моделей для Samsung Galaxy Tab S8 Pro (Snapdragon 8 Gen 1, Android 15).

## 🚀 Швидкий старт

### VPN Сервіси
```bash
# Запуск всіх VPN сервісів
./manager.sh start

# Перевірка статусу
./manager.sh status

# Зупинка сервісів
./manager.sh stop
```

### AI Моделі (Інтерактивне меню)
```bash
# Запуск головного лаунчера
./ai_launcher.sh
```

**Доступні моделі:**
- 🤖 Gemma 2B - швидкий чат (~20 tok/s, 1.6GB)
- 🤖 Gemma 9B - якісний чат (~5 tok/s, 5.8GB) ⚠️
- 🇺🇦 Ukrainian MPNet Q8 - швидкі ембеддинги (290MB)
- 🇺🇦 Ukrainian MPNet F16 - точні ембеддинги (538MB)

## 📋 Компоненти системи

### 🔐 Smart Proxy (`smart_proxy.py`)
- **SOCKS5 проксі**: порт 1080
- **HTTP проксі**: порт 8888+ (автопідбір)
- **Швейцарські заголовки**: для імітації трафіку з Швейцарії
- **Обхід VPN детекції**: спеціальні техніки маскування

### 🤖 Survey Automation (`survey_automation.py`)
- **Автоматизація опитувань**: порт 8080
- **Інтеграція з проксі**: використовує smart_proxy для анонімності

### 🛠 Manager (`manager.sh`)
- **Управління сервісами**: start/stop/restart/status
- **Логування**: централізовані логи
- **Моніторинг**: перевірка статусу процесів

---

## 🤖 AI Моделі

### 🎯 AI Launcher (`ai_launcher.sh`)
- **Інтерактивне меню**: вибір моделі через зручний інтерфейс
- **Моніторинг**: температура CPU, RAM usage
- **Логування**: централізовані логи AI моделей

### 💬 Gemma 2 (Google) - Чат-моделі
- **Gemma 2B Q4_K_M** (1.6GB) - Швидкий чат
  - Швидкість: ~15-25 tok/s
  - RAM: ~2-3GB
  - CPU: 6 потоків (A510 + A710)

- **Gemma 9B Q4_K_M** (5.8GB) - Якісний чат ⚠️
  - Швидкість: ~3-6 tok/s
  - RAM: ~6-7GB (ВАЖКА!)
  - CPU: 7 потоків (всі крім X2)

### 🇺🇦 Ukrainian MPNet - Ембеддинги
- **Q8_0** (290MB) - Швидкі ембеддинги
  - Dimension: 768
  - RAM: ~350MB
  - HTTP API: порт 8765

- **F16** (538MB) - Точні ембеддинги
  - Dimension: 768
  - RAM: ~600MB
  - HTTP API: порт 8765

### 📦 Допоміжні скрипти
- `install_embeddings.sh` - Встановлення Ukrainian MPNet
- `start_embedding_service.sh` - HTTP сервер ембеддингів
- `test_embedding_uk.sh` - Тестування з українськими текстами

## 🔧 Налаштування

### Автозапуск при старті Termux
Додайте до `~/.bashrc`:
```bash
# Автозапуск VPN сервісів
if [ -f "$HOME/vpn/manager.sh" ]; then
    echo "🚀 Запуск VPN сервісів..."
    cd "$HOME/vpn" && ./manager.sh start
fi
```

### Тестування

#### VPN
```bash
# Перевірка IP через проксі
curl --proxy socks5://127.0.0.1:1080 https://ipapi.co/json/

# Перевірка HTTP проксі
curl --proxy http://127.0.0.1:8888 https://ipapi.co/json/
```

#### AI Моделі
```bash
# Тестування ембеддингів
./test_embedding_uk.sh

# Генерація ембеддингу через API
curl -X POST http://127.0.0.1:8765/embed \
  -H "Content-Type: application/json" \
  -d '{"text":"Привіт, світ!"}'

# Health check
curl http://127.0.0.1:8765/health
```

## 📊 Моніторинг

### VPN
```bash
# Перегляд логів
./manager.sh logs proxy   # Логи проксі
./manager.sh logs survey  # Логи автоматизації

# Статус системи
./manager.sh status
```

### AI Моделі
```bash
# Моніторинг через лаунчер
./ai_launcher.sh  # Вибери опцію 6 (Моніторинг температури)

# Статус ембеддинг сервісу
./start_embedding_service.sh status

# Перегляд логів
tail -f ~/models/ukr-mpnet/service.log
tail -f ~/models/logs/*.log
```

**⚠️ Важливо для Snapdragon 8 Gen 1:**
- ✅ <60°C - норма
- ⚠️ 60-65°C - увага, можливий троттлінг
- 🔥 >65°C - зупини модель негайно!

## 🌍 Особливості

- **Швейцарська геолокація**: імітація трафіку з Швейцарії
- **Множинні протоколи**: SOCKS5 + HTTP підтримка
- **Автоматичне відновлення**: перезапуск при збоях
- **Логування**: детальні логи всіх операцій

## 📁 Структура файлів

```
~/vpn/
├── 🎯 AI МОДЕЛІ
│   ├── ai_launcher.sh                 # Головний AI лаунчер (СТАРТ ТУТ!)
│   ├── install_embeddings.sh          # Встановлення ембеддингів
│   ├── start_embedding_service.sh     # HTTP сервер ембеддингів
│   └── test_embedding_uk.sh           # Тестування ембеддингів
│
├── 🌐 VPN СЕРВІСИ
│   ├── manager.sh                     # VPN менеджер
│   ├── smart_proxy.py                 # SOCKS5/HTTP проксі
│   ├── survey_automation.py           # Автоматизація опитувань
│   ├── webrtc_block.js               # WebRTC блокування
│   ├── proxy.log / proxy.pid         # VPN логи/PID
│   └── survey.log / survey.pid       # Survey логи/PID
│
├── 📄 ДОКУМЕНТАЦІЯ
│   ├── README.md                      # Цей файл
│   └── old_files_backup_*.tar.gz     # Резервні копії
│
└── .claude/                           # Claude Code конфігурація

~/models/
├── gemma2/
│   ├── gemma-2-2b-it-Q4_K_M.gguf     (1.6GB) - швидкий чат
│   └── gemma-2-9b-it-Q4_K_M.gguf     (5.8GB) - якісний чат
│
├── embeddings/
│   ├── ukr-paraphrase-*-Q8_0.gguf    (290MB) - швидкі ембеддинги
│   └── ukr-paraphrase-*-F16.gguf     (538MB) - точні ембеддинги
│
├── ukr-mpnet/
│   ├── install_report.txt             # Звіт про встановлення
│   ├── service.log                    # Лог сервісу
│   └── test_outputs/                  # Результати тестів
│
├── logs/                              # Логи AI моделей
└── models_index.json                  # Індекс встановлених моделей

~/.local/opt/gguf/embeddings/
├── lang-uk-mpnet-Q8.gguf  -> ~/models/embeddings/...
└── lang-uk-mpnet-F16.gguf -> ~/models/embeddings/...

~/llama.cpp/
├── llama-cli                          # CLI для чату
├── llama-embedding                    # CLI для ембеддингів
└── build/                             # Зібрані бінарники
```
