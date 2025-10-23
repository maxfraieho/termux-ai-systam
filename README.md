# AI Service - Embeddings та Генеративна Модель

Система для роботи з AI моделями на Android в Termux:
- Ukrainian MPNet Embeddings (Q8_0 та F16)
- Gemma 3N генеративна модель
- HTTP API для ембеддингів
- Інтерактивний лаунчер

## 📦 Встановлені моделі

### Ukrainian MPNet Embeddings
- **Q8_0** (290MB) - швидкі ембеддинги для українських текстів
- **F16** (538MB) - точні ембеддинги для українських текстів
- Розмірність: 768
- Модель: `ukr-paraphrase-multilingual-mpnet-base`

### Gemma 3N
- **Gemma 3N Q4_K_M** (1.1GB) - генеративна модель для чату
- Квантизація: Q4_K_M (оптимальний баланс розміру та якості)

## 🚀 Швидкий старт

### 1. Запуск Embedding сервісу

```bash
# Запустити сервіс (Q8_0 варіант)
./start_embedding_service.sh start

# Запустити з F16 варіантом
VARIANT=F16 ./start_embedding_service.sh start

# Зупинити сервіс
./start_embedding_service.sh stop

# Перезапустити
./start_embedding_service.sh restart

# Статус
./start_embedding_service.sh status
```

### 2. Тестування Embeddings

```bash
# Запустити тестування
./test_embedding_uk.sh
```

### 3. AI Лаунчер (інтерактивне меню)

```bash
./ai_launcher.sh
```

Меню опцій:
1. Gemma 3N - генеративний чат
2. Ukrainian MPNet Q8 - швидкі ембеддинги
3. Ukrainian MPNet F16 - точні ембеддинги
4. Тестування ембеддингів
5. Моніторинг температури CPU
6. Перегляд логів

### 4. MD to Embeddings Service

```bash
# Інтерактивний режим
python3 md_to_embeddings_service_v4.py
```

Функції:
- Розгортання шаблону проєкту
- Конвертація DRAKON схем (`.json → .md`)
- Створення узагальнюючого `.md` з коду проєкту
- Копіювання файлів до Dropbox

## 📡 HTTP API

### Health Check
```bash
curl http://127.0.0.1:8765/health
```

Відповідь:
```json
{"status": "healthy", "model": "lang-uk-mpnet-Q8.gguf"}
```

### Генерація Embedding
```bash
curl -X POST http://127.0.0.1:8765/embed \
  -H "Content-Type: application/json" \
  -d '{"text":"Київ — столиця України."}'
```

Відповідь:
```json
{
  "embedding": [0.145, 0.015, 0.707, ...],
  "dim": 768,
  "model": "lang-uk-mpnet-Q8.gguf"
}
```

## 📁 Структура файлів

```
ai_service/
├── README_AI_SERVICE.md          # Цей файл
├── README.md                      # Оригінальний README
├── SUMMARY.md                     # Детальна документація
├── AI_INTEGRATION_SUMMARY.txt    # Звіт про інтеграцію
│
├── md_to_embeddings_service_v4.py  # Основний сервіс обробки MD
│
├── ai_launcher.sh                 # Головний інтерактивний лаунчер
├── install_embeddings.sh          # Інсталяція Ukrainian MPNet
├── start_embedding_service.sh     # Запуск HTTP сервера ембеддингів
├── start_gemma_service.sh         # Запуск Gemma генеративної моделі
├── run_md_service.sh              # Допоміжний скрипт запуску
└── test_embedding_uk.sh           # Тестування українських текстів
```

## ⚙️ Конфігурація

### Embedding Service

- **Порт**: 8765 (змінити: `PORT=8080 ./start_embedding_service.sh start`)
- **Хост**: 127.0.0.1 (локальний доступ)
- **Потоки**: 6 CPU threads
- **CPU Affinity**: 0-6
- **Модель**: Q8 або F16 (змінити: `VARIANT=F16`)

### Моделі

Моделі зберігаються в:
- `~/.local/opt/gguf/embeddings/` - Ukrainian MPNet моделі
- `~/models/` - Усі моделі та логи
- `~/models/ukr-mpnet/` - Логи та результати тестів
- `~/models/gemma3n/` - Gemma модель

## 🧪 Приклади використання

### Python

```python
import requests

# Генерація ембеддингу
response = requests.post('http://127.0.0.1:8765/embed',
    json={'text': 'Штучний інтелект в Україні'})

embedding = response.json()['embedding']
dimension = response.json()['dim']

print(f"Розмірність: {dimension}")
print(f"Перші 5 значень: {embedding[:5]}")
```

### Bash

```bash
# Отримати ембеддинг
TEXT="Машинне навчання"
curl -s -X POST http://127.0.0.1:8765/embed \
  -H "Content-Type: application/json" \
  -d "{\"text\":\"$TEXT\"}" | jq '.embedding[0:5]'
```

### JavaScript (Node.js)

```javascript
const response = await fetch('http://127.0.0.1:8765/embed', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ text: 'Київський політехнічний інститут' })
});

const data = await response.json();
console.log('Dimension:', data.dim);
console.log('Embedding:', data.embedding.slice(0, 5));
```

## 📊 Результати тестування

### Embedding тести

```
✓ Тест 1: Короткий текст
  Розмірність: 768
  Перші 3 значення: [0.145, 0.015, 0.707]

✓ Тест 2: Середній текст
  Розмірність: 768

✓ Тест 3: Семантична подібність
  Cosine similarity: 0.7516
  ✓ Подібні тексти мають високу схожість
```

Результати зберігаються в: `~/models/ukr-mpnet/test_outputs/`

## 🔧 Troubleshooting

### Сервіс не запускається

```bash
# Перевірити чи порт вільний
netstat -tlnp | grep 8765

# Подивитися логи
tail -f ~/models/ukr-mpnet/service.log

# Вбити процес якщо завис
pkill -f llama-embedding-server
```

### Модель не знайдена

```bash
# Перевірити symlinks
ls -lh ~/.local/opt/gguf/embeddings/

# Перевірити файли
ls -lh ~/models/embeddings/*.gguf

# Переінсталювати (якщо потрібно)
./install_embeddings.sh
```

### Повільна генерація

```bash
# Використати Q8_0 замість F16
VARIANT=Q8 ./start_embedding_service.sh start

# Зменшити кількість потоків
# Відредагувати THREADS в start_embedding_service.sh
```

## 📝 Логи та моніторинг

### Логи сервісу

```bash
# Переглянути логи embedding сервісу
tail -f ~/models/ukr-mpnet/service.log

# Переглянути всі логи моделей
ls ~/models/logs/
```

### Моніторинг CPU

```bash
# В AI лаунчері вибрати опцію 5
./ai_launcher.sh

# Або використати termux-sensor
termux-sensor -s android.sensor.cpu_temperature
```

## 🎯 Наступні кроки

1. **Створити Git репозиторій**:
```bash
cd ~/ai_service
git init
git add .
git commit -m "Initial commit: AI Service with Ukrainian MPNet and Gemma"
```

2. **Додати віддалений репозиторій** (GitHub/GitLab)

3. **Розширити функціонал**:
   - Додати batch processing для ембеддингів
   - Інтеграція з vector database (ChromaDB, Pinecone)
   - Web UI для ембеддингів
   - RAG (Retrieval-Augmented Generation) з Gemma

## 📚 Додаткова документація

- `AI_INTEGRATION_SUMMARY.txt` - Повний звіт про інтеграцію
- `SUMMARY.md` - Детальна документація всіх компонентів
- `README.md` - Оригінальний README проєкту

## 🆘 Підтримка

Логи помилок зберігаються в:
- `~/models/ukr-mpnet/service.log` - Embedding service
- `~/models/logs/` - Gemma та інші моделі

## 📄 Ліцензія

Моделі:
- Ukrainian MPNet: Apache 2.0
- Gemma: Gemma Terms of Use

Код: MIT (якщо не вказано інше)
