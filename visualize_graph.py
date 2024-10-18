import matplotlib.pyplot as plt
import networkx as nx

def visualize_graph():
    G = nx.Graph()
    G.add_edge('Diabetic', 'Regular Exercise', recommendation='30 mins daily')
    G.add_edge('Diabetic', 'Diet Control', recommendation='Low sugar, balanced diet')
    G.add_edge('Not Diabetic', 'Maintain Weight', recommendation='Regular weight checks')

    pos = nx.spring_layout(G)
    labels = nx.get_edge_attributes(G, 'recommendation')

    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, font_weight='bold')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.title("Knowledge Graph: Diabetes Management Recommendations")
    plt.show()

visualize_graph()
