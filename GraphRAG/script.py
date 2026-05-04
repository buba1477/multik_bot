import json
import requests
import os
import re
from collections import defaultdict

# Настройки файлов
INPUT_FILE = "test100.jsonl"
OUTPUT_FILE = "graph_nodes.jsonl"
OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "yagpt5_4096:latest"

# Лимиты
MAX_ENTITIES_PER_CHUNK = 12
MAX_RELATIONS_PER_CHUNK = 15

def call_ollama_ai(prompt):
    """Вызов локальной модели через Ollama API"""
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
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
        text = text.strip()
        # Убираем маркдаун код-блоки
        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)
        
        # Ищем JSON объект
        start = text.find('{')
        end = text.rfind('}') + 1
        if start != -1 and end != 0:
            return json.loads(text[start:end])
        return {"entities": [], "relationships": []}
    except Exception as e:
        print(f"⚠️ Ошибка парсинга JSON: {e}")
        return {"entities": [], "relationships": []}

def normalize_entity_name(name):
    """Нормализует имя сущности для дедупликации"""
    name = name.strip()
    # Убираем кавычки разных типов
    name = re.sub(r'["«»“”\']', '', name)
    # Приводим к нижнему регистру
    name = name.lower()
    # Убираем множественное число в конце
    name = re.sub(r'ы$|и$|а$|я$', '', name)
    # Схлопываем пробелы
    name = re.sub(r'\s+', ' ', name)
    return name

def get_universal_prompt(chunk_text, doc_type=None):
    """
    Универсальный промт для любого документа
    doc_type: необязательный хинт (например "закон", "инструкция", "техническая документация")
    """
    type_hint = ""
    if doc_type:
        type_hint = f"Тип документа: {doc_type}.\n"
    
    return (
        f"Ты — эксперт по извлечению знаний из текстов.\n"
        f"{type_hint}"
        f"Проанализируй фрагмент документа и извлеки из него ключевые сущности и связи.\n\n"
        f"ТЕКСТ ДЛЯ АНАЛИЗА:\n{chunk_text[:3000]}\n\n"
        f"ПРАВИЛА:\n"
        f"1. Сущности — это ВАЖНЫЕ объекты: документы, статьи, люди, организации, термины, понятия, даты, события.\n"
        f"2. НЕ вытаскивай общие слова ('текст', 'документ', 'часть', 'пункт' без конкретики).\n"
        f"3. НЕ вытаскивай номера страниц, ID чанков, техническую информацию.\n"
        f"4. Для каждой сущности напиши КРАТКОЕ описание (2-5 слов или короткая фраза).\n"
        f"5. Связи — это глаголы или короткие фразы ('является', 'включает', 'запрещает', 'определяет', 'ссылается на').\n"
        f"6. Если связь двусторонняя — создай одну связь с глаголом.\n"
        f"7. Ограничения: не более {MAX_ENTITIES_PER_CHUNK} сущностей и {MAX_RELATIONS_PER_CHUNK} связей.\n\n"
        f"ФОРМАТ ОТВЕТА (ТОЛЬКО JSON, БЕЗ ПОЯСНЕНИЙ):\n"
        f"{{\n"
        f"  \"entities\": [\n"
        f"    {{\"name\": \"Название сущности\", \"description\": \"Краткое описание\"}}\n"
        f"  ],\n"
        f"  \"relationships\": [\n"
        f"    {{\"source\": \"Сущность A\", \"target\": \"Сущность B\", \"relation\": \"тип связи\"}}\n"
        f"  ]\n"
        f"}}\n\n"
        f"JSON:"
    )

def detect_document_type(chunk_text):
    """Пытается угадать тип документа по первым строкам"""
    if re.search(r'79-ФЗ|Федеральный закон|статья', chunk_text, re.IGNORECASE):
        return "законодательство"
    elif re.search(r'инструкция|руководство|порядок|регламент', chunk_text, re.IGNORECASE):
        return "инструкция"
    elif re.search(r'отчет|анализ|результаты|исследование', chunk_text, re.IGNORECASE):
        return "отчет"
    elif re.search(r'ТУ|ГОСТ|СНиП|стандарт|технические условия', chunk_text, re.IGNORECASE):
        return "техническая документация"
    else:
        return None

def process_pipeline():
    if not os.path.exists(INPUT_FILE):
        print(f"❌ Файл {INPUT_FILE} не найден!")
        return

    print(f"🚀 Запуск универсальной графовой экстракции. Модель: {MODEL_NAME}")
    
    # Глобальные хранилища
    all_entities = {}
    all_relationships = []
    entity_counter = 1
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as f_in, \
         open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        
        for i, line in enumerate(f_in):
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                chunk_id = data.get('id', f'chunk_{i+1}')
                chunk_text = data.get('text', '')
                
                if not chunk_text:
                    continue
                
                # Определяем тип документа (хинт для модели)
                doc_type = detect_document_type(chunk_text)
                
                # Генерируем промт
                prompt = get_universal_prompt(chunk_text[:3000], doc_type)
                
                raw_response = call_ollama_ai(prompt)
                
                if not raw_response:
                    print(f"⚠️ Чанк {i+1}: пустой ответ от модели")
                    data['graph_data'] = {"entities": [], "relationships": [], "error": "empty_response"}
                    f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
                    continue
                
                extracted = clean_and_parse_json(raw_response)
                
                # Обработка сущностей с дедупликацией
                local_entities = []
                for ent in extracted.get('entities', [])[:MAX_ENTITIES_PER_CHUNK]:
                    name = ent.get('name', '').strip()
                    description = ent.get('description', '').strip()
                    
                    if not name or len(name) < 3:
                        continue
                    
                    # Фильтруем мусорные сущности
                    if name.lower() in ['текст', 'документ', 'страница', 'пункт', 'часть', 'раздел']:
                        continue
                    
                    norm_name = normalize_entity_name(name)
                    
                    if norm_name not in all_entities:
                        all_entities[norm_name] = {
                            "id": f"ent_{entity_counter:04d}",
                            "name": name,
                            "description": description,
                            "freq": 1,
                            "chunks": [chunk_id]
                        }
                        entity_counter += 1
                    else:
                        all_entities[norm_name]["freq"] += 1
                        if chunk_id not in all_entities[norm_name]["chunks"]:
                            all_entities[norm_name]["chunks"].append(chunk_id)
                        # Обновляем описание, если оно длиннее
                        if len(description) > len(all_entities[norm_name].get("description", "")):
                            all_entities[norm_name]["description"] = description
                    
                    local_entities.append({
                        "name": name,
                        "description": description,
                        "normalized": norm_name
                    })
                
                # Обработка связей
                local_relations = []
                for rel in extracted.get('relationships', [])[:MAX_RELATIONS_PER_CHUNK]:
                    source = rel.get('source', '').strip()
                    target = rel.get('target', '').strip()
                    relation = rel.get('relation', 'связана с').strip()
                    
                    if not source or not target:
                        continue
                    
                    # Нормализуем source и target для дедупликации связей
                    norm_source = normalize_entity_name(source)
                    norm_target = normalize_entity_name(target)
                    
                    local_relations.append({
                        "source": source,
                        "target": target,
                        "relation": relation
                    })
                    
                    # Проверяем, есть ли уже такая связь
                    existing = False
                    for r in all_relationships:
                        if (r.get("source_norm") == norm_source and 
                            r.get("target_norm") == norm_target and
                            r.get("relation") == relation):
                            r["weight"] = r.get("weight", 1) + 1
                            existing = True
                            break
                    
                    if not existing:
                        all_relationships.append({
                            "source": source,
                            "target": target,
                            "source_norm": norm_source,
                            "target_norm": norm_target,
                            "relation": relation,
                            "weight": 1,
                            "chunks": [chunk_id]
                        })
                    else:
                        # Добавляем чанк к существующей связи
                        for r in all_relationships:
                            if (r.get("source_norm") == norm_source and 
                                r.get("target_norm") == norm_target and
                                r.get("relation") == relation):
                                if chunk_id not in r.get("chunks", []):
                                    r.setdefault("chunks", []).append(chunk_id)
                                break
                
                # Сохраняем результат для этого чанка
                data['graph_data'] = {
                    "entities": local_entities,
                    "relationships": local_relations,
                    "doc_type_hint": doc_type,
                    "stats": {
                        "total_entities_global": len(all_entities),
                        "entities_in_chunk": len(local_entities),
                        "relations_in_chunk": len(local_relations)
                    }
                }
                f_out.write(json.dumps(data, ensure_ascii=False) + "\n")
                
                print(f"✅ Чанк {i+1} ({data.get('id', 'N/A')[:40]}): "
                      f"{len(local_entities)} сущностей, {len(local_relations)} связей")
                
            except json.JSONDecodeError as e:
                print(f"❌ Ошибка JSON на строке {i+1}: {e}")
            except Exception as e:
                print(f"⚠️ Общая ошибка на строке {i+1}: {e}")
    
    # Сохраняем глобальный граф
    print(f"\n📊 СТАТИСТИКА ГРАФА:")
    print(f"   - Уникальных сущностей: {len(all_entities)}")
    print(f"   - Уникальных связей: {len(all_relationships)}")
    
    # Сортируем сущности по частоте
    sorted_entities = sorted(all_entities.values(), key=lambda x: x.get('freq', 0), reverse=True)
    
    with open("graph_global.json", 'w', encoding='utf-8') as f_g:
        json.dump({
            "entities": sorted_entities,
            "relationships": all_relationships,
            "stats": {
                "total_entities": len(all_entities),
                "total_relationships": len(all_relationships),
                "model": MODEL_NAME
            }
        }, f_g, ensure_ascii=False, indent=2)
    
    # Выводим топ-10 сущностей
    print(f"\n🏆 ТОП-10 сущностей по частоте:")
    for i, ent in enumerate(sorted_entities[:10]):
        print(f"   {i+1}. {ent['name']} (freq: {ent['freq']})")
    
    print(f"\n🏁 Готово! Результаты сохранены:")
    print(f"   - Чанки с графами: {OUTPUT_FILE}")
    print(f"   - Глобальный граф: graph_global.json")

if __name__ == "__main__":
    process_pipeline()