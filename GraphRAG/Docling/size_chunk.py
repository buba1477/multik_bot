import json
from transformers import AutoTokenizer

# Путь к модели и файлу
MODEL_PATH = "/home/amlin04/multik_bot/hf_cache/multilingual-e5-large"
JSONL_FILE = "test_size.jsonl"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

def check_base():
    max_tokens = 0
    over_limit_count = 0
    total_chunks = 0
    fat_chunks = []

    print(f"🧐 Начинаю проверку файла: {JSONL_FILE}...")

    with open(JSONL_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            total_chunks += 1
            data = json.loads(line)
            text = data.get("text", "")
            id = data.get("id", "N/A")
            
            # Считаем честные токены
            tokens_count = len(tokenizer.encode(text))
            
            if tokens_count > max_tokens:
                max_tokens = tokens_count
            
            if tokens_count > 512:
                over_limit_count += 1
                fat_chunks.append((id, tokens_count))

    print("\n" + "="*40)
    print(f"📊 ИТОГИ ПРОВЕРКИ:")
    print(f"✅ Всего чанков: {total_chunks}")
    print(f"🔥 Самый жирный чанк: {max_tokens} токенов")
    print(f"❌ Чанков больше 512 лимита: {over_limit_count}")
    print("="*40)

    if fat_chunks:
        print("\n🚨 СПИСОК НАРУШИТЕЛЕЙ (ID и размер):")
        # Сортируем по размеру и выводим топ-10
        fat_chunks.sort(key=lambda x: x[1], reverse=True)
        for id, count in fat_chunks[:10]:
            print(f"   - {id}: {count} токенов")

if __name__ == "__main__":
    check_base()
