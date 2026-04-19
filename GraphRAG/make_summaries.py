import json
import requests

CLUSTERS_FILE = "communities.json" # Твой файл с кластерами
OUTPUT_SUMMARIES = "graph_summaries.jsonl"
OLLAMA_API_URL = "http://localhost:11434/api/generate"

def get_summary(nodes):
    prompt = (
        f"Ты — эксперт-аналитик. Перед тобой список сущностей из раздела закона №79-ФЗ:\n"
        f"{', '.join(nodes)}\n"
        f"ЗАДАЧА: Напиши ОДНИМ абзацем (3-4 предложения), о чем этот блок и какая его главная цель.\n"
        f"Пиши строго по делу. ОТВЕТ:"
    )
    res = requests.post(OLLAMA_API_URL, json={
        "model": "yagpt5_4096:latest",
        "prompt": prompt,
        "stream": False
    })
    return res.json().get('response', '')

with open(CLUSTERS_FILE, 'r', encoding='utf-8') as f:
    communities = json.load(f)

with open(OUTPUT_SUMMARIES, 'w', encoding='utf-8') as out:
    for cid, nodes in communities.items():
        print(f"📝 Обработка сообщества {cid}...")
        summary = get_summary(nodes)
        result = {
            "id": f"comm_{cid}",
            "text": f"ОБЗОР РАЗДЕЛА: {summary}",
            "metadata": {"type": "community_summary", "nodes": nodes}
        }
        out.write(json.dumps(result, ensure_ascii=False) + "\n")

print("✅ ВСЁ! Теперь у тебя есть 'Глобальные знания' о законе.")
