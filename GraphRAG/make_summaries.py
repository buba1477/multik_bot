import json
import requests
import os

CLUSTERS_FILE = "communities.json"
OUTPUT_SUMMARIES = "graph_summaries.jsonl"
OLLAMA_API_URL = "http://localhost:11434/api/generate"

def summarize_community_v2(nodes_info):
    # СОБИРАЕМ ПОДРОБНЫЙ КОНТЕКСТ ДЛЯ ИИ
    context_lines = []
    for n in nodes_info:
        # Теперь n — это словарь {'name': ..., 'description': ...}
        line = f"- {n['name']}: {n['description']}"
        context_lines.append(line)
    
    context_str = "\n".join(context_lines)
    
    # ЖЕСТКИЙ ПРОМПТ ДЛЯ ГЛОБАЛЬНОГО ПОНИМАНИЯ
    prompt = (
        f"Ты — ведущий юрист ФНС. Перед тобой справочник понятий из раздела закона №79-ФЗ.\n"
        f"СПРАВОЧНИК РАЗДЕЛА:\n{context_str}\n\n"
        f"ЗАДАЧА: Напиши ПОДРОБНОЕ аналитическое резюме этого раздела (4-6 предложений).\n"
        f"1. Какова главная цель этого блока?\n"
        f"2. Какие ключевые объекты и изменения (законы) здесь важны?\n"
        f"3. Как эти понятия связаны между собой в системе госслужбы?\n"
        f"Пиши профессионально и емко. ОТВЕТ:"
    )
    
    payload = {
        "model": "yagpt5_4096:latest",
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1}
    }
    
    try:
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=300)
        return response.json().get('response', '')
    except Exception as e:
        return f"Ошибка суммаризации: {e}"

def run():
    if not os.path.exists(CLUSTERS_FILE):
        print(f"❌ Файл {CLUSTERS_FILE} не найден!")
        return

    with open(CLUSTERS_FILE, 'r', encoding='utf-8') as f:
        communities = json.load(f)

    with open(OUTPUT_SUMMARIES, 'w', encoding='utf-8') as out:
        for cid, nodes_info in communities.items():
            print(f"📝 Анализируем сообщество {cid} ({len(nodes_info)} сущностей)...")
            
            summary = summarize_community_v2(nodes_info)
            
            # Сохраняем расширенный объект для поиска
            result = {
                "id": f"community_{cid}",
                "title": f"Аналитический обзор раздела {cid}",
                "text": summary,
                "metadata": {
                    "type": "community_summary",
                    # Сохраняем только имена в метаданные для краткости
                    "nodes": [n['name'] for n in nodes_info]
                }
            }
            out.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"🏁 ГОТОВО! Проверь {OUTPUT_SUMMARIES}. Теперь это уровень Microsoft GraphRAG.")

if __name__ == "__main__":
    run()
