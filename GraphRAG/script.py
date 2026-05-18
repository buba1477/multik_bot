import json
import requests
import os
import re

# ========================================
# CONFIG
# ========================================

INPUT_FILE = "test100.jsonl"
OUTPUT_FILE = "graph_nodes.jsonl"

OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "yagpt5_4096:latest"

MAX_ENTITIES_PER_CHUNK = 12
MAX_RELATIONS_PER_CHUNK = 15

# ========================================
# GRAPH RULES
# ========================================

ALLOWED_RELATIONS = [
    "регулирует",
    "определяет",
    "содержит",
    "ссылается на",
    "запрещает",
    "разрешает",
    "требует",
    "устанавливает",
    "включает",
    "применяется к",
    "связан с"
]

ENTITY_ALIASES = {
    "конкурсный отбор": "конкурс",
    "проведение конкурса": "конкурс",
    "конкурс на замещение": "конкурс",

    "гражданский служащий": "госслужащий",
    "гражданские служащие": "госслужащий",

    "федеральная налоговая служба": "фнс",
    "фнс россии": "фнс",
}

BAD_ENTITIES = [
    "текст",
    "документ",
    "страница",
    "пункт",
    "часть",
    "раздел",
    "данные",
    "информация",
    "система",
    "орган",
    "лицо",
    "гражданин",
    "служба",
    "деятельность",
    "порядок"
]

# ========================================
# OLLAMA
# ========================================

def call_ollama_ai(prompt):

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
        response = requests.post(
            OLLAMA_API_URL,
            json=payload,
            timeout=300
        )

        response.raise_for_status()

        return response.json().get("response", "")

    except Exception as e:
        print(f"❌ Ошибка API: {e}")
        return ""

# ========================================
# CLEAN JSON
# ========================================

def clean_and_parse_json(text):

    try:
        text = text.strip()

        text = re.sub(r'```json\s*', '', text)
        text = re.sub(r'```\s*', '', text)

        start = text.find('{')
        end = text.rfind('}') + 1

        if start != -1 and end != 0:
            return json.loads(text[start:end])

        return {
            "entities": [],
            "relationships": []
        }

    except Exception as e:
        print(f"⚠️ Ошибка парсинга JSON: {e}")

        return {
            "entities": [],
            "relationships": []
        }

# ========================================
# NORMALIZATION
# ========================================

def normalize_entity_name(name):

    name = name.strip()

    # убрать кавычки
    name = re.sub(r'["«»“”\']', '', name)

    # lowercase
    name = name.lower()

    # пробелы
    name = re.sub(r'\s+', ' ', name)

    # aliases
    if name in ENTITY_ALIASES:
        name = ENTITY_ALIASES[name]

    return name

# ========================================
# PROMPT
# ========================================

def get_universal_prompt(chunk_text, doc_type=None):

    type_hint = ""

    if doc_type:
        type_hint = f"Тип документа: {doc_type}.\n"

    relations_text = "\n".join(
        [f"- {r}" for r in ALLOWED_RELATIONS]
    )

    return (
        f"Ты — система извлечения знаний из документов.\n"
        f"{type_hint}\n"

        f"Извлеки только явно указанные в тексте сущности и связи.\n\n"

        f"ТЕКСТ:\n"
        f"{chunk_text[:3000]}\n\n"

        f"ПРАВИЛА:\n"

        f"1. Сущности:\n"
        f"- законы\n"
        f"- статьи\n"
        f"- приказы\n"
        f"- организации\n"
        f"- юридические понятия\n"
        f"- процедуры\n"
        f"- сроки\n\n"

        f"2. НЕ извлекать:\n"
        f"- общие слова\n"
        f"- технический мусор\n"
        f"- служебные слова\n\n"

        f"3. Связь должна быть ЯВНО в тексте.\n"
        f"4. НЕ придумывай связи.\n"

        f"5. Используй ТОЛЬКО связи из списка:\n"
        f"{relations_text}\n\n"

        f"6. Максимум:\n"
        f"- {MAX_ENTITIES_PER_CHUNK} сущностей\n"
        f"- {MAX_RELATIONS_PER_CHUNK} связей\n\n"

        f"ФОРМАТ:\n"

        f"{{\n"
        f'  "entities": [\n'
        f'    {{"name":"сущность","description":"описание"}}\n'
        f"  ],\n"
        f'  "relationships": [\n'
        f'    {{"source":"A","target":"B","relation":"регулирует"}}\n'
        f"  ]\n"
        f"}}\n\n"

        f"ТОЛЬКО JSON:"
    )

# ========================================
# DOCUMENT TYPE
# ========================================

def detect_document_type(chunk_text):

    if re.search(r'79-ФЗ|Федеральный закон|статья', chunk_text, re.IGNORECASE):
        return "законодательство"

    elif re.search(r'инструкция|руководство|порядок|регламент', chunk_text, re.IGNORECASE):
        return "инструкция"

    elif re.search(r'отчет|анализ|результаты|исследование', chunk_text, re.IGNORECASE):
        return "отчет"

    elif re.search(r'ТУ|ГОСТ|СНиП|стандарт|технические условия', chunk_text, re.IGNORECASE):
        return "техническая документация"

    return None

# ========================================
# MAIN PIPELINE
# ========================================

def process_pipeline():

    if not os.path.exists(INPUT_FILE):
        print(f"❌ Файл {INPUT_FILE} не найден!")
        return

    print(f"🚀 Запуск GraphRAG extraction")

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

                doc_type = detect_document_type(chunk_text)

                prompt = get_universal_prompt(
                    chunk_text[:3000],
                    doc_type
                )

                raw_response = call_ollama_ai(prompt)

                if not raw_response:

                    data['graph_data'] = {
                        "entities": [],
                        "relationships": [],
                        "error": "empty_response"
                    }

                    f_out.write(
                        json.dumps(data, ensure_ascii=False) + "\n"
                    )

                    continue

                extracted = clean_and_parse_json(raw_response)

                local_entities = []

                # ========================================
                # ENTITIES
                # ========================================

                for ent in extracted.get('entities', [])[:MAX_ENTITIES_PER_CHUNK]:

                    name = ent.get('name', '').strip()
                    description = ent.get('description', '').strip()

                    if not name or len(name) < 3:
                        continue

                    norm_name = normalize_entity_name(name)

                    if norm_name in BAD_ENTITIES:
                        continue

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

                        if len(description) > len(
                            all_entities[norm_name].get("description", "")
                        ):
                            all_entities[norm_name]["description"] = description

                    local_entities.append({
                        "name": name,
                        "description": description,
                        "normalized": norm_name
                    })

                # ========================================
                # RELATIONS
                # ========================================

                local_relations = []

                for rel in extracted.get('relationships', [])[:MAX_RELATIONS_PER_CHUNK]:

                    source = rel.get('source', '').strip()
                    target = rel.get('target', '').strip()

                    relation = rel.get(
                        'relation',
                        'связан с'
                    ).strip()

                    if relation not in ALLOWED_RELATIONS:
                        continue

                    if not source or not target:
                        continue

                    norm_source = normalize_entity_name(source)
                    norm_target = normalize_entity_name(target)

                    local_relations.append({
                        "source": source,
                        "target": target,
                        "relation": relation
                    })

                    existing = False

                    for r in all_relationships:

                        if (
                            r.get("source_norm") == norm_source and
                            r.get("target_norm") == norm_target and
                            r.get("relation") == relation
                        ):

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

                # ========================================
                # SAVE CHUNK
                # ========================================

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

                f_out.write(
                    json.dumps(data, ensure_ascii=False) + "\n"
                )

                print(
                    f"✅ {chunk_id} | "
                    f"{len(local_entities)} entities | "
                    f"{len(local_relations)} relations"
                )

            except Exception as e:
                print(f"⚠️ Ошибка: {e}")

    # ========================================
    # SAVE GLOBAL GRAPH
    # ========================================

    sorted_entities = sorted(
        all_entities.values(),
        key=lambda x: x.get('freq', 0),
        reverse=True
    )

    with open(
        "graph_global.json",
        'w',
        encoding='utf-8'
    ) as f_g:

        json.dump({
            "entities": sorted_entities,
            "relationships": all_relationships,
            "stats": {
                "total_entities": len(all_entities),
                "total_relationships": len(all_relationships),
                "model": MODEL_NAME
            }
        }, f_g, ensure_ascii=False, indent=2)

    print("\n==============================")
    print("✅ GRAPH EXTRACTION FINISHED")
    print(f"Entities: {len(all_entities)}")
    print(f"Relations: {len(all_relationships)}")
    print("==============================")

# ========================================
# START
# ========================================

if __name__ == "__main__":
    process_pipeline()