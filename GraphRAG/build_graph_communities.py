import json
import networkx as nx
import community as community_louvain
import os

# Твой новый файл с экстракцией (где есть поле description)
INPUT_FILE = "graph_nodes.jsonl"

def build_and_partition():
    if not os.path.exists(INPUT_FILE):
        print("❌ Сначала дождись финиша экстракции (graph_nodes.jsonl)!")
        return

    G = nx.Graph()
    # Хранилище описаний для каждой сущности
    descriptions = {}

    print("🔄 Сборка энциклопедической паутины...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                data = json.loads(line)
                graph = data.get('graph_data', {})
                if not isinstance(graph, dict): continue

                # 1. Обрабатываем узлы и сохраняем описания
                for entity in graph.get('entities', []):
                    # Если ИИ вернул словарь (как мы договорились для v2)
                    if isinstance(entity, dict):
                        name = entity.get('name')
                        desc = entity.get('description', '')
                        if name:
                            G.add_node(name)
                            # Сохраняем самое длинное описание (чтобы было максимум инфы)
                            if len(desc) > len(descriptions.get(name, '')):
                                descriptions[name] = desc
                    else:
                        # На случай, если проскочила старая строка
                        G.add_node(str(entity))

                # 2. Добавляем связи
                for rel in graph.get('relationships', []):
                    src = rel.get('source')
                    tgt = rel.get('target')
                    if src and tgt:
                        G.add_edge(src, tgt, relation=rel.get('relation', 'связан'))
            except Exception as e:
                print(f"⚠️ Ошибка на строке {i+1}: {e}")

    if G.number_of_nodes() == 0:
        print("❌ Граф пуст! Проверь содержимое graph_nodes.jsonl")
        return

    print(f"📊 Граф собран: {G.number_of_nodes()} узлов, {G.number_of_edges()} связей.")

    # 3. Алгоритм Лувена (Community Detection)
    print("🧠 Группируем смыслы (Louvain)...")
    # Нужно установить: pip install python-louvain
    partition = community_louvain.best_partition(G)
    
    # Собираем данные для каждой группы
    communities = {}
    for node, comm_id in partition.items():
        node_info = {
            "name": node,
            "description": descriptions.get(node, "Описание не извлечено")
        }
        communities.setdefault(comm_id, []).append(node_info)

    print(f"✅ Найдено сообществ: {len(communities)}")
    
    # Сохраняем расширенные данные для нашего УМНОГО суммаризатора
    with open("communities.json", "w", encoding="utf-8") as f:
        json.dump(communities, f, ensure_ascii=False, indent=2)
    
    print(f"💾 Данные сохранены в communities.json")
    return communities

if __name__ == "__main__":
    build_and_partition()
