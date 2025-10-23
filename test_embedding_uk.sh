#!/data/data/com.termux/files/usr/bin/bash
################################################################################
# Тестування Ukrainian MPNet Embedding моделі
################################################################################

set -euo pipefail

SERVICE_URL="${SERVICE_URL:-http://127.0.0.1:8765}"
OUTPUT_DIR="$HOME/models/ukr-mpnet/test_outputs"

mkdir -p "$OUTPUT_DIR"

# Кольори
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${CYAN}🧪 Тестування Ukrainian MPNet Embedding моделі${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Перевірка сервісу
echo -n "Перевірка сервісу... "
if curl -s --max-time 2 "$SERVICE_URL/health" >/dev/null 2>&1; then
    echo -e "${GREEN}✓ OK${NC}"
else
    echo -e "${YELLOW}✗ Недоступний${NC}"
    echo ""
    echo "Запусти сервіс: ./start_embedding_service.sh start"
    exit 1
fi

# Тестові тексти
TEXT1="Київ — столиця України."
TEXT2="Штучний інтелект швидко розвивається."
TEXT3="Сьогодні гарна погода."

echo ""
echo -e "${CYAN}Тест 1: Короткий текст${NC}"
echo "Текст: $TEXT1"
echo -n "Генерація ембеддингу... "

RESULT1=$(curl -s -X POST "$SERVICE_URL/embed" \
    -H "Content-Type: application/json; charset=utf-8" \
    -d @- <<JSON
{"text":"$TEXT1"}
JSON
)

if echo "$RESULT1" | jq -e '.embedding' >/dev/null 2>&1; then
    DIM=$(echo "$RESULT1" | jq '.dim')
    SAMPLE=$(echo "$RESULT1" | jq -r '.embedding[0:3] | @json')
    echo -e "${GREEN}✓ OK${NC}"
    echo "  Розмірність: $DIM"
    echo "  Перші 3 значення: $SAMPLE"
    echo "$RESULT1" > "$OUTPUT_DIR/test1.json"
else
    echo -e "${YELLOW}✗ Помилка${NC}"
fi

echo ""
echo -e "${CYAN}Тест 2: Середній текст${NC}"
echo "Текст: $TEXT2"
echo -n "Генерація ембеддингу... "

RESULT2=$(curl -s -X POST "$SERVICE_URL/embed" \
    -H "Content-Type: application/json" \
    -d "{\"text\":\"$TEXT2\"}")

if echo "$RESULT2" | jq -e '.embedding' >/dev/null 2>&1; then
    DIM=$(echo "$RESULT2" | jq '.dim')
    echo -e "${GREEN}✓ OK${NC}"
    echo "  Розмірність: $DIM"
    echo "$RESULT2" > "$OUTPUT_DIR/test2.json"
else
    echo -e "${YELLOW}✗ Помилка${NC}"
fi

echo ""
echo -e "${CYAN}Тест 3: Семантична подібність${NC}"
echo "Текст A: $TEXT2"
echo "Текст B: $TEXT3"

RESULT3=$(curl -s -X POST "$SERVICE_URL/embed" \
    -H "Content-Type: application/json" \
    -d "{\"text\":\"$TEXT3\"}")

if echo "$RESULT3" | jq -e '.embedding' >/dev/null 2>&1; then
    echo "$RESULT3" > "$OUTPUT_DIR/test3.json"

    # Розрахунок cosine similarity (спрощений)
    python3 - <<PYTHON
import json, math

with open('$OUTPUT_DIR/test2.json') as f:
    emb2 = json.load(f)['embedding']

with open('$OUTPUT_DIR/test3.json') as f:
    emb3 = json.load(f)['embedding']

dot = sum(a*b for a,b in zip(emb2, emb3))
norm2 = math.sqrt(sum(a*a for a in emb2))
norm3 = math.sqrt(sum(b*b for b in emb3))

similarity = dot / (norm2 * norm3) if norm2 and norm3 else 0

print(f"Cosine similarity: {similarity:.4f}")

if similarity > 0.5:
    print("✓ Подібні тексти мають високу схожість")
else:
    print("✓ Різні тексти мають низьку схожість")
PYTHON

fi

echo ""
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}✓ Тестування завершено${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "Результати збережено в: $OUTPUT_DIR"
echo ""
