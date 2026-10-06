import json
import os
import time
import re
from dotenv import load_dotenv
load_dotenv("/home/amlin04/multik_bot/.env")

CLUSTERS_FILE = "communities.json"
OUTPUT_SUMMARIES = "graph_summaries.jsonl"

# GigaChat
GIGACHAT_KEY = os.getenv("API_KEY_GIGACHAT", "")
GIGACHAT_MODEL = "GigaChat-3-Ultra"

MAX_NODES_PER_CLUSTER = 12
DELAY_BETWEEN_REQUESTS = 1.5
MAX_RETRIES = 2

def summarize_community(community_id, nodes_info):
    """Формирует глубокий контекст с описаниями и жесткими правилами"""
    
    # Сортируем по важности и берем топ
    sorted_nodes = sorted(nodes_info, key=lambda x: x.get('degree', 0), reverse=True)
    nodes_to_process = sorted_nodes[:MAX_NODES_PER_CLUSTER]
    
    # Формируем список: Имя [Тип] - Описание
    context_lines = []
    for n in nodes_to_process:
        name = n.get('name', '')
        n_type = n.get('type', 'concept')
        desc = n.get('description', 'описание отсутствует')
        context_lines.append(f"- {name} [{n_type}]: {desc}")
    
    context_str = "\n".join(context_lines)
    
    # 🔥 ЖЕСТКИЙ ПРОМПТ С ЯКОРЕМ ВРЕМЕНИ
    prompt = f"""Ты — главный аналитик правового управления ФНС России. 
Твоя задача: на основе списка сущностей описать их логическую связь в рамках СОВРЕМЕННОГО законодательства (79-ФЗ, приказы ФНС).

⚠️ ЗАПРЕТЫ:
1. КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО упоминать "МНС" или "Министерство налогов и сборов". Этой структуры нет 20 лет.
2. Не используй исторический контекст. Только текущие реалии ФНС России.
3. Не пиши фразы "Этот кластер объединяет..." или "В данный список входят...". Пиши сразу суть.

СУЩНОСТИ ДЛЯ АНАЛИЗА:
{context_str}

Кратко перечисли:
- какие темы связаны
- какие сущности чаще всего встречаются вместе
- какие нормы регулируют один процесс

НЕ делай выводов.
НЕ интерпретируй. """
    
    return prompt

def call_gigachat(prompt, max_retries=MAX_RETRIES):
    if not GIGACHAT_KEY:
        print("❌ API_KEY_GIGACHAT не задан! Пропускаем.")
        return "Связи между объектами определяются регламентами ФНС России и положениями 79-ФЗ."

    from gigachat import GigaChat as GigaChatSDK

    payload = {
        "model": GIGACHAT_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 4096,
    }

    for attempt in range(max_retries):
        try:
            with GigaChatSDK(credentials=GIGACHAT_KEY, verify_ssl_certs=False) as giga:
                response = giga.chat(payload)
                result = response.choices[0].message.content.strip()

                # 🔥 ФИЗИЧЕСКАЯ ЗАЧИСТКА ГАЛЛЮЦИНАЦИЙ (регистронезависимая)
                bad_words = [
                    r'министерств[а-я\s]+налогов\s+и\s+сборов', 
                    r'\bМНС\b', 
                    r'\bМНС\s+России\b'
                ]
                for pattern in bad_words:
                    result = re.sub(pattern, 'ФНС России', result, flags=re.IGNORECASE)

                if len(result) > 15:
                    return result
        except Exception as e:
            print(f"   ⚠️ Ошибка GigaChat (попытка {attempt+1}): {e}")
        
        time.sleep(3)
    
    return "Связи между объектами определяются регламентами ФНС России и положениями 79-ФЗ."

def run():
    if not os.path.exists(CLUSTERS_FILE):
        print(f"❌ Файл {CLUSTERS_FILE} не найден!")
        return

    with open(CLUSTERS_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    communities = data.get('communities', {})
    processed = 0
    
    with open(OUTPUT_SUMMARIES, 'w', encoding='utf-8') as out:
        for cid, nodes_info in communities.items():
            if len(nodes_info) < 2: 
                continue
            
            print(f"📝 Анализ сообщества {cid} ({len(nodes_info)} узлов)...")
            prompt = summarize_community(cid, nodes_info)
            summary = call_gigachat(prompt)
            
            # Чистим вводные фразы, если модель их всё-таки добавила
            summary = re.sub(r'^(Аналитический вывод:|Вывод:|Данный кластер|Этот набор)\s*', '', summary)

            result = {
                "id": f"community_{cid}",
                "title": f"Аналитический обзор кластера {cid}",
                "text": summary,
                "metadata": {
                    "type": "community_summary",
                    "num_nodes": len(nodes_info),
                    "nodes": [n.get('name') for n in nodes_info[:15]]
                }
            }
            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            processed += 1
            print(f"   📊 Готово: {len(summary)} симв.")
            time.sleep(DELAY_BETWEEN_REQUESTS)
    
    print(f"\n🏁 Завершено. Обработано {processed} резюме.")

if __name__ == "__main__":
    run()
