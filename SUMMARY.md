# 📋 SUMMARY - AI Models & VPN Infrastructure

**Останнє оновлення:** 12 жовтня 2025 (вечір) - v1.1
**Платформа:** Samsung Galaxy Tab S8 Pro (SM-X906B)
**CPU:** Snapdragon 8 Gen 1 (aarch64)
**Android:** 15
**Середовище:** Termux

---

## ✅ ЩО ВСТАНОВЛЕНО

### 🤖 AI Infrastructure

#### llama.cpp (Зібрано успішно)
- **Розташування:** `~/llama.cpp/`
- **Бінарники:** `~/llama.cpp/build/bin/`
  - `llama-cli` (2.5MB) - для чату
  - `llama-embedding` (2.5MB) - для ембеддингів
  - `llama-server` (4.6MB) - HTTP сервер
- **Статус:** ✅ Готовий до використання

#### Ukrainian MPNet Embeddings (Встановлено)
- **Q8_0** (290MB) - швидкі ембеддинги
  - Шлях: `~/models/embeddings/ukr-paraphrase-multilingual-mpnet-base-Q8_0.gguf`
  - Симлінк: `~/.local/opt/gguf/embeddings/lang-uk-mpnet-Q8.gguf`
  - SHA256: `b2681e224043f0a675ea1c5e00c1f5f1a405d04048ef8d2814446b914d07516e`

- **F16** (538MB) - точні ембеддинги
  - Шлях: `~/models/embeddings/ukr-paraphrase-multilingual-mpnet-base-F16.gguf`
  - Симлінк: `~/.local/opt/gguf/embeddings/lang-uk-mpnet-F16.gguf`
  - SHA256: `c51b469ddb71f93c67116ecfd1ff16b4bfc71e5c88c38953d7433b859d5a5ca0`

- **HTTP Сервіс:** Працює на порту 8765
- **Статус:** ✅ Працює з українським текстом

#### Gemma Models (Потребують завантаження)
- **Gemma 2B Q4_K_M** (1.6GB) - швидкий чат
  - Очікуваний шлях: `~/models/gemma2/gemma-2-2b-it-Q4_K_M.gguf`
  - Статус: ⚠️ Не завантажено

- **Gemma 9B Q4_K_M** (5.8GB) - якісний чат
  - Очікуваний шлях: `~/models/gemma2/gemma-2-9b-it-Q4_K_M.gguf`
  - Статус: ⚠️ Не завантажено

### 🌐 VPN Services

- **Smart Proxy** (`smart_proxy.py`) - SOCKS5/HTTP проксі
- **Survey Automation** (`survey_automation.py`) - автоматизація
- **Manager** (`manager.sh`) - управління VPN
- **Статус:** ✅ Працює незалежно від AI

---

## 🚀 ШВИДКИЙ СТАРТ

### Запуск AI Launcher (Рекомендовано)

```bash
cd ~/vpn
./ai_launcher.sh
```

**Меню:**
1. Gemma 2B - швидкий чат
2. Gemma 9B - якісний чат
3. Ukrainian MPNet Q8 - швидкі ембеддинги ✅
4. Ukrainian MPNet F16 - точні ембеддинги ✅
5. Тестування ембеддингів ✅
6. Моніторинг температури CPU
7. Перегляд логів
8. Зупинити всі AI моделі ✅ (БЕЗПЕЧНО для VPN!)

### Ембеддинг сервіс (Окремо)

```bash
# Запуск
./start_embedding_service.sh start --variant Q8   # або F16

# Статус
./start_embedding_service.sh status

# Зупинка
./start_embedding_service.sh stop

# Перезапуск
./start_embedding_service.sh restart
```

### VPN сервіси

```bash
cd ~/vpn
./manager.sh start    # Запуск VPN
./manager.sh status   # Статус
./manager.sh stop     # Зупинка
```

---

## 🔄 АВТОМАТИЧНА ЗУПИНКА МОДЕЛЕЙ (НОВА ФУНКЦІЯ!)

**Оновлення від 12.10.2025 (вечір):**

### ✅ Реалізовано:

1. **Автоматична зупинка при запуску нової моделі**
   - Коли запускаєш будь-яку модель (Gemma 2B/9B, MPNet Q8/F16)
   - Автоматично зупиняються всі інші AI процеси
   - **ГАРАНТІЯ:** VPN процеси (`smart_proxy.py`, `survey_automation.py`) НЕ зачіпаються!

2. **Ручна зупинка всіх AI моделей**
   - Опція 8 в головному меню: "🛑 Зупинити всі AI моделі"
   - Зупиняє: `llama-cli`, `llama-server`, ембеддинг сервіси
   - **ЗАХИСТ:** Потрійний фільтр grep виключає VPN процеси

### 🛡️ Захист VPN сервісів:

```bash
# Функція get_running_models() має подвійний захист:
ps aux | grep -E 'llama-cli|python3.*embed|llama-server' | \
    grep -v grep | \
    grep -v smart_proxy | \          # Виключення VPN SOCKS5 проксі
    grep -v survey_automation         # Виключення VPN автоматизації

# Функція kill_all_models() має потрійний захист:
# 1. Фільтрація при пошуку процесів
# 2. Фільтрація при kill
# 3. Фільтрація при kill -9 (примусове завершення)
```

### Як працює:

**Приклад 1:** Запуск моделі з автозупинкою інших
```bash
./ai_launcher.sh
# Вибери опцію 1 (Gemma 2B)
# Система автоматично:
#   ✓ Знайде запущені AI процеси
#   ✓ Покаже які саме (PID, назва)
#   ✓ Зупинить їх
#   ✓ НЕ зачепить smart_proxy.py та survey_automation.py
#   ✓ Запустить Gemma 2B
```

**Приклад 2:** Ручна зупинка всіх моделей
```bash
./ai_launcher.sh
# Вибери опцію 8
# Система:
#   ✓ Покаже скільки процесів знайдено
#   ✓ Зупинить всі AI моделі
#   ✓ VPN продовжить працювати
```

### Перевірка безпеки VPN:

```bash
# Перевір що VPN працює ДО зупинки AI
ps aux | grep -E 'smart_proxy|survey_automation' | grep -v grep

# Зупини AI моделі (опція 8 в меню)

# Перевір що VPN ДОСІ працює ПІСЛЯ зупинки AI
ps aux | grep -E 'smart_proxy|survey_automation' | grep -v grep
# Має показати ті самі процеси з тими самими PID!
```

### Модифіковані функції в ai_launcher.sh:

1. `start_gemma_2b()` - додано `kill_all_models()` на початку
2. `start_gemma_9b()` - додано `kill_all_models()` на початку
3. `start_mpnet_q8()` - додано `kill_all_models()` на початку
4. `start_mpnet_f16()` - додано `kill_all_models()` на початку
5. Нова опція меню "8) Зупинити всі AI моделі"

**Файли змінено:**
- `~/vpn/ai_launcher.sh` (оновлено)
- `~/vpn/SUMMARY.md` (цей файл, додано звіт)

---

## 📁 СТРУКТУРА ПРОЕКТУ

```
~/vpn/
├── 🎯 AI SCRIPTS
│   ├── ai_launcher.sh              # Головний лаунчер (СТАРТ ТУТ!)
│   ├── install_embeddings.sh       # Встановлення ембеддингів
│   ├── start_embedding_service.sh  # HTTP сервер ембеддингів
│   └── test_embedding_uk.sh        # Тестування з українським текстом
│
├── 🌐 VPN SCRIPTS
│   ├── manager.sh                  # VPN менеджер
│   ├── smart_proxy.py              # SOCKS5/HTTP проксі
│   ├── survey_automation.py        # Автоматизація
│   └── webrtc_block.js            # WebRTC блокування
│
├── 📄 DOCUMENTATION
│   ├── README.md                   # Повна документація
│   ├── SUMMARY.md                  # Цей файл
│   ├── AI_INTEGRATION_SUMMARY.txt  # Звіт інтеграції
│   └── old_files_backup_*.tar.gz  # Резервні копії
│
└── .claude/                        # Claude Code конфігурація

~/models/
├── embeddings/
│   ├── ukr-paraphrase-*-Q8_0.gguf  ✅
│   └── ukr-paraphrase-*-F16.gguf   ✅
│
├── gemma2/
│   ├── (gemma-2-2b-it-Q4_K_M.gguf)  ⚠️ Не завантажено
│   └── (gemma-2-9b-it-Q4_K_M.gguf)  ⚠️ Не завантажено
│
├── ukr-mpnet/
│   ├── install_report.txt          # Звіт встановлення
│   ├── service.log                 # Лог сервісу
│   ├── service.pid                 # PID файл
│   └── test_outputs/               # Результати тестів
│
├── logs/                           # AI логи
└── models_index.json               # Індекс моделей

~/.local/opt/gguf/embeddings/
├── lang-uk-mpnet-Q8.gguf  -> ~/models/embeddings/...
└── lang-uk-mpnet-F16.gguf -> ~/models/embeddings/...

~/llama.cpp/
├── build/bin/
│   ├── llama-cli         ✅
│   ├── llama-embedding   ✅
│   └── llama-server      ✅
└── (весь репозиторій llama.cpp)
```

---

## 💡 ПРИКЛАДИ ВИКОРИСТАННЯ

### 1. Тестування Ukrainian MPNet

```bash
cd ~/vpn

# Запусти сервіс (якщо не запущено)
./start_embedding_service.sh start --variant Q8

# Запусти тести
./test_embedding_uk.sh
```

**Очікуваний результат:**
```
✓ OK
Розмірність: 768
Cosine similarity: 0.7283
```

### 2. Генерація ембеддингу через API

**Bash:**
```bash
echo '{"text":"Київ — столиця України"}' | \
  curl -X POST http://127.0.0.1:8765/embed \
  -H 'Content-Type: application/json' \
  -d @- | jq '.embedding | length'
# Вивід: 768
```

**Python:**
```python
import requests

response = requests.post(
    'http://127.0.0.1:8765/embed',
    json={'text': 'Штучний інтелект змінює світ'}
)

embedding = response.json()['embedding']
print(f"Dimension: {len(embedding)}")  # 768
print(f"First 5 values: {embedding[:5]}")
```

### 3. Завантаження Gemma моделей (якщо потрібно)

```bash
# Створи теку
mkdir -p ~/models/gemma2

# Gemma 2B (1.6GB - швидкий, ~5-10 хвилин)
wget https://huggingface.co/bartowski/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it-Q4_K_M.gguf \
  -P ~/models/gemma2/

# Gemma 9B (5.8GB - якісний, ~20-30 хвилин)
wget https://huggingface.co/bartowski/gemma-2-9b-it-GGUF/resolve/main/gemma-2-9b-it-Q4_K_M.gguf \
  -P ~/models/gemma2/
```

Після завантаження вони автоматично з'являться в `ai_launcher.sh`.

---

## 🔧 ВАЖЛИВІ КОМАНДИ

### Перевірка статусу

```bash
# Ембеддинг сервіс
./start_embedding_service.sh status

# VPN
./manager.sh status

# Процеси
ps aux | grep -E 'python3|llama-cli'

# Порти
netstat -tuln 2>/dev/null | grep -E '8765|1080|8888' || echo "netstat недоступний"
```

### Логи

```bash
# Ембеддинг сервіс
tail -f ~/models/ukr-mpnet/service.log

# VPN
tail -f ~/vpn/proxy.log
tail -f ~/vpn/survey.log

# Всі AI логи
ls ~/models/logs/
```

### Моніторинг системи

```bash
# RAM
free -h

# CPU температура
cat /sys/class/thermal/thermal_zone*/temp | awk '{print $1/1000 "°C"}'

# Використання диску
df -h ~

# Процеси по CPU
top -bn1 | head -20
```

---

## ⚠️ ВАЖЛИВІ ОБМЕЖЕННЯ (Snapdragon 8 Gen 1)

### Температура CPU
- ✅ **<60°C** - норма
- ⚠️ **60-65°C** - увага, можливий троттлінг
- 🔥 **>65°C** - ЗУПИНИ МОДЕЛЬ негайно!

**Як моніторити:**
```bash
./ai_launcher.sh  # Опція 6 (Моніторинг температури)
```

### Використання RAM

| Модель | RAM | Рекомендація |
|--------|-----|--------------|
| Ukrainian MPNet Q8 | ~350MB | ✅ Завжди OK |
| Ukrainian MPNet F16 | ~600MB | ✅ OK |
| Gemma 2B | ~2-3GB | ✅ OK |
| Gemma 9B | ~6-7GB | ⚠️ Закрий інші додатки! |

**Доступна RAM:** 7GB (з 12GB) при чистій системі

### CPU Threading

Моделі оптимізовані для Snapdragon 8 Gen 1:
- **4x Cortex-A510** (1.78 GHz) - енергоефективні
- **3x Cortex-A710** (2.49 GHz) - продуктивні
- **1x Cortex-X2** (2.99 GHz) - PRIME ядро

**Конфігурація:**
- Ukrainian MPNet: 6 потоків (CPU 0-5)
- Gemma 2B: 6 потоків (CPU 0-5)
- Gemma 9B: 7 потоків (CPU 0-6)
- **CPU 7 (X2) - НЕ використовується** (залишено для системи, уникнення перегріву)

---

## 🐛 TROUBLESHOOTING

### Проблема: Сервіс не запускається

**Помилка:** `Address already in use`

**Рішення:**
```bash
# Знайди старий процес
ps aux | grep python3 | grep -v grep

# Зупини (замість XXXX підставь PID)
kill XXXX

# Або через скрипт
./start_embedding_service.sh stop

# Запусти заново
./start_embedding_service.sh start
```

### Проблема: llama-cli не знайдено

**Помилка:** `llama-cli не знайдено`

**Рішення:**
```bash
# Перевір чи зібрано
ls ~/llama.cpp/build/bin/llama-cli

# Якщо немає - збери заново
cd ~/llama.cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc)
```

### Проблема: Модель не знайдена

**Помилка:** `Model not found`

**Рішення:**
```bash
# Перевір наявність
ls -lh ~/models/embeddings/*.gguf
ls -lh ~/models/gemma2/*.gguf

# Якщо Ukrainian MPNet відсутній
cd ~/vpn
./install_embeddings.sh

# Якщо Gemma відсутній - див. секцію "Завантаження Gemma"
```

### Проблема: Українські символи не працюють

**Помилка:** `Invalid \escape` або неправильне відображення

**Рішення:**
- ✅ Використовуй `test_embedding_uk.sh` - він правильно обробляє UTF-8
- ✅ В Python використовуй `requests.post(..., json={'text': '...'})` - автоматично UTF-8
- ⚠️ В curl використовуй heredoc:
  ```bash
  curl -X POST http://127.0.0.1:8765/embed \
    -H 'Content-Type: application/json' \
    -d @- <<JSON
  {"text":"Український текст"}
  JSON
  ```

### Проблема: Gemma модель дуже повільна

**Симптом:** Gemma 9B генерує <2 tok/s

**Причини та рішення:**
1. **Перегрів:** Подивись температуру (опція 6 в меню)
2. **Swap:** Закрий інші додатки, звільни RAM
3. **Використовуй Gemma 2B:** Набагато швидша (~20 tok/s)

---

## 📚 РЕСУРСИ

### Документація
- **README.md** - повна документація
- **AI_INTEGRATION_SUMMARY.txt** - детальний звіт
- **SUMMARY.md** - цей файл

### Онлайн ресурси
- [llama.cpp GitHub](https://github.com/ggerganov/llama.cpp)
- [Ukrainian MPNet HuggingFace](https://huggingface.co/podarok/ukr-paraphrase-multilingual-mpnet-base)
- [Gemma Models](https://huggingface.co/bartowski)

### Звіти
- Встановлення: `~/models/ukr-mpnet/install_report.txt`
- Тестування: `~/models/ukr-mpnet/test_outputs/test_report.txt`
- Індекс моделей: `~/models/models_index.json`

---

## 🔄 ЩО ДАЛІ

### Якщо потрібно завантажити Gemma

1. Вибери модель:
   - **Gemma 2B** - для швидкої роботи (рекомендовано)
   - **Gemma 9B** - для якісних відповідей (повільно)

2. Завантаж (див. розділ "Завантаження Gemma моделей")

3. Запусти через `ai_launcher.sh` - модель автоматично з'явиться в меню

### Для інтеграції в свої проекти

**Python приклад:**
```python
import requests

class UkrainianEmbeddings:
    def __init__(self, url="http://127.0.0.1:8765"):
        self.url = f"{url}/embed"

    def embed(self, text):
        """Генерує 768-вимірний ембеддинг"""
        response = requests.post(self.url, json={'text': text})
        return response.json()['embedding']

    def similarity(self, text1, text2):
        """Обчислює cosine similarity"""
        import numpy as np

        emb1 = np.array(self.embed(text1))
        emb2 = np.array(self.embed(text2))

        return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

# Використання
embedder = UkrainianEmbeddings()
sim = embedder.similarity(
    "Штучний інтелект",
    "Машинне навчання"
)
print(f"Подібність: {sim:.4f}")
```

---

## 📞 ШВИДКА ДОВІДКА

```bash
# Запуск
cd ~/vpn && ./ai_launcher.sh

# Ембеддинги (готово ✅)
./start_embedding_service.sh start --variant Q8
./test_embedding_uk.sh

# VPN (працює незалежно ✅)
./manager.sh start

# Моніторинг
./ai_launcher.sh  # Опція 6

# Логи
tail -f ~/models/ukr-mpnet/service.log

# Статус
./start_embedding_service.sh status
ps aux | grep python3
```

---

**Версія:** 1.1
**Дата створення:** 12.10.2025
**Останнє оновлення:** 12.10.2025 (вечір) - додано автозупинку AI моделей
**Автор:** Автоматично згенеровано Claude Code

**Все працює! Готово до використання! 🚀**

### 📝 Історія змін:

- **v1.1** (12.10.2025 вечір):
  - ✅ Додано автоматичну зупинку AI моделей при запуску нової
  - ✅ Додано опцію ручної зупинки всіх моделей (опція 8 в меню)
  - ✅ Потрійний захист VPN сервісів від випадкового завершення
  - ✅ Відображення запущених процесів у статусі меню

- **v1.0** (12.10.2025):
  - Початкова версія
  - Встановлення Ukrainian MPNet Q8/F16
  - Інтеграція з Gemma 2B/9B
  - Створення ai_launcher.sh
