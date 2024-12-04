import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx


def visualize_graph(data: Data, seed: int = 7):

    """
    Visualize graph using networkx and matplotlib

    Parameters
    ----------
    data: torch_geometric.data.Data
        Graph data

    Example
    -------
    >>> visualize_graph(data)
    """

    G = to_networkx(data, to_undirected=True)
    plt.figure(figsize=(7, 7))
    plt.xticks([])
    plt.yticks([])
    nx.draw_networkx(G, pos=nx.spring_layout(G, seed=seed), with_labels=True, node_color=data.y, cmap="Set2", node_size=25)
    plt.show()
