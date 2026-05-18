import json
from transformers import AutoTokenizer

# Путь к модели и файлу
MODEL_PATH = "/home/amlin04/multik_bot/hf_cache/ru-en-RoSBERTa"
JSONL_FILE = "146-FZ.jsonl"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

def check_base():
    max_tokens = 0
    over_limit_count = 0
    total_chunks = 0
    fat_chunks = []
    small_chunks = [] # Список для "малышей"

    print(f"🧐 Начинаю проверку файла: {JSONL_FILE}...")

    with open(JSONL_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            total_chunks += 1
            data = json.loads(line)
            text = data.get("text", "")
            node_id = data.get("id", "N/A")
            
            # Считаем честные токены
            tokens_count = len(tokenizer.encode(text))
            
            if tokens_count > max_tokens:
                max_tokens = tokens_count
            
            # Проверка на превышение лимита
            if tokens_count > 512:
                over_limit_count += 1
                fat_chunks.append((node_id, tokens_count))
            
            # Проверка на слишком маленькие чанки (RED FLAG)
            if tokens_count < 150:
                small_chunks.append((node_id, tokens_count))

    print("\n" + "="*40)
    print(f"📊 ИТОГИ ПРОВЕРКИ:")
    print(f"✅ Всего чанков: {total_chunks}")
    print(f"🔥 Самый жирный чанк: {max_tokens} токенов")
    print(f"❌ Чанков больше 512 лимита: {over_limit_count}")
    print(f"🚩 Чанков меньше 150 (small): {len(small_chunks)}")
    print("="*40)

    if fat_chunks:
        print("\n🚨 СПИСОК НАРУШИТЕЛЕЙ (>512 токенов):")
        fat_chunks.sort(key=lambda x: x[1], reverse=True)
        for node_id, count in fat_chunks:
            print(f"   - {node_id}: {count} токенов")

    if small_chunks:
        print("\n⚠️ СПИСОК МАЛЫШЕЙ (<150 токенов):")
        # Сортируем от самых крошечных
        small_chunks.sort(key=lambda x: x[1])
        for node_id, count in small_chunks: # Показываем первые 15
            print(f"   - {node_id}: {count} токенов")
        

if __name__ == "__main__":
    check_base()
