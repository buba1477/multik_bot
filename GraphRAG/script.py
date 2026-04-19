import json
import requests
import os

# Настройки файлов
INPUT_FILE = "test100.jsonl"
OUTPUT_FILE = "graph_nodes.jsonl"
OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "yagpt5_4096:latest"

def call_ollama_ai(prompt):
    """Вызов локальной модели через Ollama API"""
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0, # Ставим 0 для максимальной строгости
            "num_ctx": 4096
        }
    }
    try:
        response = requests.post(OLLAMA_API_URL, json=payload, timeout=300)
        response.raise_for_status()
        return response.json().get('response', '')
    except Exception as e:
        print(f"❌ Ошибка API: {e}")
        return ""

def clean_and_parse_json(text):
    """Чистит ответ модели от мусора и превращает в объект"""
    try:
        text = text.strip().replace('```json', '').replace('```', '')
        start = text.find('{')
        end = text.rfind('}') + 1
        if start != -1 and end != 0:
            return json.loads(text[start:end])
        return {"error": "no_json_found", "raw": text}
    except:
        return {"error": "parse_error", "raw": text}

def process_pipeline():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Файл {INPUT_FILE} не найден!")
        return

    print(f"🚀 Стартуем СТРОГУЮ экстракцию. Модель: {MODEL_NAME}")

    with open(INPUT_FILE, 'r', encoding='utf-8') as f_in, \
         open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        
        for i, line in enumerate(f_in):
            line = line.strip()
            if not line: continue
            
            try:
                data = json.loads(line)
                chunk_text = data.get('text', '')

                # ЖЕСТКИЙ ПРОМПТ: Указываем ключи и запрещаем списки в target
                prompt = (
                    f"ИНСТРУКЦИЯ: Извлеки сущности и связи из текста.\n"
                    f"Верни ответ СТРОГО в формате JSON.\n"
                    f"КЛЮЧИ: 'entities' (список строк), 'relationships' (список объектов).\n"
                    f"ФОРМАТ СВЯЗИ: {{\"source\": \"A\", \"relation\": \"делает\", \"target\": \"B\"}}.\n"
                    f"ПРАВИЛО: Если у одного source несколько target, создай ОТДЕЛЬНЫЙ объект для каждой связи.\n"
                    f"ТЕКСТ ДЛЯ АНАЛИЗА:\n{chunk_text}\n\n"
                    f"JSON:"
                )

                raw_response = call_ollama_ai(prompt)
                data['graph_data'] = clean_and_parse_json(raw_response)
                
                f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
                print(f"✅ Чанк {i+1} (ID: {data.get('id', 'N/A')}) — OK")

            except Exception as e:
                print(f"⚠️ Ошибка на строке {i+1}: {e}")

    print(f"🏁 Готово! Теперь ключи всегда: source, relation, target.")

if __name__ == "__main__":
    process_pipeline()
