import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

def plot_ticker_graph(G: nx.DiGraph, ticker: str):

    """
    Plot the graph of a ticker and its neighbors
    
    Parameters
    ----------
    G : nx.Graph
        The graph to plot
        
    ticker : str
        The ticker to plot
        
    Raises
    ------
    ValueError
        If the ticker is not found in the dataset
    
    Examples
    --------
    >>> plot_ticker_graph(G, 'AAPL')
    """
    try:
        node_index = [node_tuple[0] for node_tuple in list(G.nodes.data("name")) if node_tuple[1] == ticker][0]
    except IndexError:
        raise ValueError(f"Ticker {ticker} not found in the dataset")
    
    plt.figure(figsize=(10, 10))
    subgraph = G.subgraph([node_index] + list(G.neighbors(node_index)))
    nx.draw_networkx(subgraph, pos=nx.spring_layout(subgraph, seed=7), labels = nx.get_node_attributes(subgraph, 'name'),with_labels=True, cmap="Set2")
    plt.show()


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
