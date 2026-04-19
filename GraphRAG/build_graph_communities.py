import json
import networkx as nx
import community as community_louvain
import os

INPUT_FILE = "graph_nodes.jsonl"

def build_and_partition():
    if not os.path.exists(INPUT_FILE):
        print("❌ Сначала дождись финиша экстракции!")
        return

    G = nx.Graph()

    print("🔄 Сборка глобальной паутины...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            graph = data.get('graph_data', {})
            
            if not isinstance(graph, dict): continue

            # Добавляем узлы
            for entity in graph.get('entities', []):
                G.add_node(entity)

            # Добавляем связи
            for rel in graph.get('relationships', []):
                src = rel.get('source')
                tgt = rel.get('target')
                if src and tgt:
                    G.add_edge(src, tgt, relation=rel.get('relation'))

    print(f"📊 Граф собран: {G.number_of_nodes()} узлов, {G.number_of_edges()} связей.")

    # КИЛЛЕР-ФИЧА: Алгоритм Лувена (Community Detection)
    print("🧠 Ищем тематические сообщества (Louvain)...")
    partition = community_louvain.best_partition(G)
    
    # Группируем узлы по сообществам для наглядности
    communities = {}
    for node, comm_id in partition.items():
        communities.setdefault(comm_id, []).append(node)

    print(f"✅ Найдено сообществ: {len(communities)}")
    
    # Выведем топ-3 сообщества для теста
    for i, (comm_id, nodes) in enumerate(list(communities.items())[:3]):
        print(f"🔹 Сообщество {comm_id}: {', '.join(nodes[:5])}...")

    # Сохраняем результат для следующего шага (Summarization)
    with open("communities.json", "w", encoding="utf-8") as f:
        json.dump(communities, f, ensure_ascii=False, indent=2)
    
    return communities

if __name__ == "__main__":
    build_and_partition()
