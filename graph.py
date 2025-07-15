import networkx as nx
import matplotlib.pyplot as plt

def visualize_kb(kb):
    G = nx.DiGraph()

    # Цвета для типов сущностей
    type_colors = {
        "Person": "#ff9999",   # светло-красный
        "Company": "#99ccff",  # светло-синий
        "Country": "#99ff99",  # светло-зеленый
        "Default": "#dddddd"   # серый для неизвестных типов
    }

    # Добавляем только основные сущности (без aliases)
    for ent_id, data in kb.items():
        node_color = type_colors.get(data.get("type"), type_colors["Default"])
        G.add_node(ent_id, label=data["name"], color=node_color)

        for linked_id in data.get("linked", []):
            if linked_id in kb:
                G.add_edge(ent_id, linked_id)

    pos = nx.spring_layout(G, seed=42, k=0.8)

    node_colors = [data["color"] for _, data in G.nodes(data=True)]
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1500)
    nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=15, edge_color='gray')

    labels = {node: data["label"] for node, data in G.nodes(data=True)}
    nx.draw_networkx_labels(G, pos, labels, font_size=10, font_weight="bold")

    plt.title("Entity Knowledge Graph", fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
