import json
import networkx as nx
import community as community_louvain
import os
from collections import Counter

INPUT_FILE = "graph_nodes.jsonl"
OUTPUT_GEPHI = "graph_export.gexf"

def build_and_partition():
    if not os.path.exists(INPUT_FILE):
        print("❌ Сначала дождись финиша экстракции (graph_nodes.jsonl)!")
        return

    G = nx.DiGraph()
    descriptions = {}
    relations_counter = Counter()
    
    print("🔄 Сборка энциклопедической паутины...")
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            try:
                data = json.loads(line)
                graph = data.get('graph_data', {})
                chunk_id = data.get('id', f'chunk_{i}')
                
                if not isinstance(graph, dict):
                    continue

                for entity in graph.get('entities', []):
                    if isinstance(entity, dict):
                        name = entity.get('name')
                        desc = entity.get('description', '')
                        if name:
                            G.add_node(name)
                            if len(desc) > len(descriptions.get(name, {}).get('text', '')):
                                descriptions[name] = {
                                    "text": desc,
                                    "first_chunk": chunk_id,
                                    "type": "unknown"
                                }
                            
                            if 'статья' in name.lower() or 'ст.' in name.lower():
                                descriptions[name]["type"] = "article"
                            elif 'закон' in name.lower() or 'фз' in name.lower():
                                descriptions[name]["type"] = "law"
                            elif 'орган' in name.lower() or 'федеральный' in name.lower():
                                descriptions[name]["type"] = "authority"
                            elif 'приказ' in name.lower() or 'постановление' in name.lower():
                                descriptions[name]["type"] = "normative_act"
                            else:
                                descriptions[name]["type"] = "concept"

                for rel in graph.get('relationships', []):
                    src = rel.get('source')
                    tgt = rel.get('target')
                    relation = rel.get('relation', 'связана с')
                    
                    if src and tgt:
                        # Считаем вес связи
                        edge_key = (src, tgt, relation)
                        relations_counter[edge_key] += 1
                        
                        # Добавляем ребро с весом
                        if G.has_edge(src, tgt):
                            current_weight = G[src][tgt].get('weight', 0)
                            current_relations = G[src][tgt].get('relations', [])
                            G[src][tgt]['weight'] = current_weight + 1
                            if relation not in current_relations:
                                current_relations.append(relation)
                            # 🔥 ПРЕОБРАЗУЕМ СПИСОК В СТРОКУ ДЛЯ GEXF
                            G[src][tgt]['relations_str'] = ', '.join(current_relations)
                            G[src][tgt]['relations'] = current_relations  # оставляем для внутреннего использования
                        else:
                            G.add_edge(src, tgt, 
                                      weight=1, 
                                      relations=[relation],
                                      relations_str=relation)  # строка для GEXF
                            
            except Exception as e:
                print(f"⚠️ Ошибка на строке {i+1}: {e}")

    if G.number_of_nodes() == 0:
        print("❌ Граф пуст! Проверь содержимое graph_nodes.jsonl")
        return

    print(f"\n📊 Граф собран:")
    print(f"   - Узлов: {G.number_of_nodes()}")
    print(f"   - Связей: {G.number_of_edges()}")
    print(f"   - Тип графа: направленный")
    
    # Вычисляем степени
    degrees = dict(G.degree())
    top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:10]
    print("\n🏆 TOP-10 узлов по степени:")
    for node, degree in top_nodes:
        node_type = descriptions.get(node, {}).get('type', 'concept')
        print(f"      {node} [{node_type}] → {degree} связей")
    
    if G.number_of_nodes() > 1:
        density = nx.density(G)
        print(f"   Плотность графа: {density:.4f}")
    
    # Кластеризация Лувена
    print("\n🧠 Группируем смыслы (Louvain)...")
    G_undirected = G.to_undirected()
    partition = community_louvain.best_partition(G_undirected)
    
    communities = {}
    comm_stats = {}
    
    for node, comm_id in partition.items():
        node_info = {
            "name": node,
            "description": descriptions.get(node, {}).get('text', "Описание не извлечено"),
            "type": descriptions.get(node, {}).get('type', 'concept'),
            "degree": degrees.get(node, 0)
        }
        communities.setdefault(comm_id, []).append(node_info)
        comm_stats[comm_id] = comm_stats.get(comm_id, 0) + 1
    
    print(f"   Найдено сообществ: {len(communities)}")
    print(f"   Размеры сообществ (топ-5): {sorted(comm_stats.values(), reverse=True)[:5]}")
    
    # Сохраняем данные
    output_data = {
        "communities": communities,
        "stats": {
            "total_nodes": G.number_of_nodes(),
            "total_edges": G.number_of_edges(),
            "density": density if G.number_of_nodes() > 1 else 0,
            "num_communities": len(communities),
            "community_sizes": comm_stats,
            "top_nodes": [{"name": n, "degree": d} for n, d in top_nodes]
        },
        "node_details": descriptions
    }
    
    with open("communities.json", "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 Данные сохранены в communities.json")
    
    # 🔥 ИСПРАВЛЕННЫЙ ЭКСПОРТ В GEXF
    # Создаём копию графа с безопасными атрибутами
    G_export = nx.DiGraph()
    
    # Добавляем узлы с атрибутами
    for node, attrs in G.nodes(data=True):
        G_export.add_node(node)
        # Добавляем тип узла, если есть
        node_type = descriptions.get(node, {}).get('type', 'concept')
        G_export.nodes[node]['type'] = node_type
        # Добавляем описание (обрезанное для GEXF)
        desc = descriptions.get(node, {}).get('text', '')
        if desc:
            G_export.nodes[node]['description'] = desc[:500]  # GEXF не любит длинные строки
    
    # Добавляем рёбра с безопасными атрибутами
    for u, v, attrs in G.edges(data=True):
        weight = attrs.get('weight', 1)
        relations_str = attrs.get('relations_str', attrs.get('relation', 'связана с'))
        G_export.add_edge(u, v, weight=weight, label=relations_str)
    
    # Экспортируем
    try:
        nx.write_gexf(G_export, OUTPUT_GEPHI)
        print(f"📁 Экспорт в Gephi: {OUTPUT_GEPHI}")
    except Exception as e:
        print(f"⚠️ Ошибка экспорта в GEXF: {e}")
        print("   (граф всё равно сохранён в communities.json)")
    
    return output_data

if __name__ == "__main__":
    result = build_and_partition()
    
    if result:
        print(f"\n🏁 ИТОГИ КЛАСТЕРИЗАЦИИ:")
        communities = result.get('communities', {})
        for comm_id, nodes in list(communities.items())[:5]:  # топ-5 кластеров
            if len(nodes) >= 3:
                print(f"   Кластер {comm_id} ({len(nodes)} узлов):")
                top_nodes_in_comm = sorted(nodes, key=lambda x: x.get('degree', 0), reverse=True)[:3]
                for n in top_nodes_in_comm:
                    print(f"      - {n['name']} [{n['type']}]")