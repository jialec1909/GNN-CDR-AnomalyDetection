from torch_geometric.utils import to_networkx
import networkx as nx
import matplotlib.pyplot as plt
import torch
import numpy as np
import os

output_dir = "./graphs_folder/graph_zones"
os.makedirs(output_dir, exist_ok=True)
def visualize_graph_customized(G, color, i):
    plt.figure(figsize=(7,7))
    plt.xticks([])
    plt.yticks([])
    nx.draw_networkx(G, pos=nx.spring_layout(G, seed=42), with_labels=False,
                     node_size = 10, node_color=color, cmap="Set2")

    filename = os.path.join(output_dir, f'graph_high_activityzones_{i}.png')
    plt.savefig(filename)
    # plt.show(block = False)
    plt.close()

def visualize_graph(data, id):

    G = to_networkx(data, to_undirected=True)
    data.y = torch.tensor(
        np.random.randint(low=1, high=2, size=(2400, ))
    )
    G = to_networkx(data, to_undirected=True)


    visualize_graph_customized(G=G, color = data.y, i = id)
