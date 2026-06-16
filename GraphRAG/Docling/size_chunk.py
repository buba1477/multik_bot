import json
from transformers import AutoTokenizer

# Путь к модели и файлу
MODEL_PATH = "/home/amlin04/multik_bot/hf_cache/FRIDA"
JSONL_FILE = "79-ФЗ.jsonl"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)


def count_full_chunk_tokens(data):
    """Считает токены с учётом префикса [ID] [TITLE] + текст"""
    doc_id = data.get("id", "").split("_st")[0]  # "79-фз"
    title = data.get("title", "")
    text = data.get("text", "")
    
    prefix = f"[{doc_id.upper()}] [{title}]"
    full_text = prefix + "\n\n" + text
    
    return len(tokenizer.encode(full_text, add_special_tokens=False))


def check_base():
    max_tokens = 0
    over_limit_count = 0
    total_chunks = 0
    fat_chunks = []
    small_chunks = []

    print(f"🧐 Начинаю проверку файла: {JSONL_FILE}...")
    print(f"📏 Учитываю префикс [ID] [TITLE] + два переноса строки\n")

    with open(JSONL_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            total_chunks += 1
            data = json.loads(line)
            node_id = data.get("id", "N/A")
            
            # 🔥 СЧИТАЕМ ПОЛНЫЙ КОНТЕКСТ (с префиксом)
            tokens_count = count_full_chunk_tokens(data)
            
            if tokens_count > max_tokens:
                max_tokens = tokens_count
            
            # Проверка на превышение жёсткого лимита
            if tokens_count > 512:
                over_limit_count += 1
                fat_chunks.append((node_id, tokens_count))
            
            # Проверка на слишком маленькие чанки
            if tokens_count < 150:
                small_chunks.append((node_id, tokens_count))

    print("\n" + "=" * 50)
    print(f"📊 ИТОГИ ПРОВЕРКИ (С ПРЕФИКСОМ):")
    print(f"✅ Всего чанков: {total_chunks}")
    print(f"🔥 Самый жирный чанк: {max_tokens} токенов")
    print(f"❌ Чанков больше 512 лимита: {over_limit_count}")
    print(f"🚩 Чанков меньше 150 (small): {len(small_chunks)}")
    print("=" * 50)

    if fat_chunks:
        print("\n🚨 СПИСОК НАРУШИТЕЛЕЙ (>512 токенов):")
        fat_chunks.sort(key=lambda x: x[1], reverse=True)
        for node_id, count in fat_chunks:
            print(f"   - {node_id}: {count} токенов")

    if small_chunks:
        print("\n⚠️ ТОП-15 САМЫХ МАЛЕНЬКИХ МАЛЫШЕЙ (<150 токенов):")
        small_chunks.sort(key=lambda x: x[1])
        for node_id, count in small_chunks[:15]:
            print(f"   - {node_id}: {count} токенов")


if __name__ == "__main__":
    check_base()