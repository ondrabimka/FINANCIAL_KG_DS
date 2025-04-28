# %%
import matplotlib.pyplot as plt
from torch_geometric.transforms import ToUndirected
from torch_geometric.utils import to_networkx

from financial_kg_ds.datasets.graph_loader import GraphLoaderBase

# %% get data
data = GraphLoaderBase.get_data()
data = ToUndirected()(data)

# %%
data
# %%
data['ticker'].name

# %% convert to networkx
G = to_networkx(data, node_attrs=['name'])

# %% make sure the number of nodes is the same
assert G.number_of_nodes() == data.num_nodes
# %% same for edges
# assert G.number_of_edges() == data.num_edges

# %%
print(G.number_of_edges())

# %%
G.edges()

# %%
G.nodes[3000].values()
# %%
dir(G)
# %%
G.edges()
# %%
list(G.edges)
# %% get all nodes connected to node 0
list(G.neighbors(0))
# %% select subgraph of nodes connected to node 0 and node 0

# %%
for keu in list(G.nodes.data("name")):
    print(keu)


# %%
len(list(G.neighbors(0)))

# %%
subgraph.nodes[0]

# %%
import networkx as nx
import matplotlib.pyplot as plt
from financial_kg_ds.utils.visualisation import plot_ticker_graph

plot_ticker_graph(G, 'AAPL')

# %%
type(G)

# %%
for i, name in enumerate(G.nodes.data("name")):
    print(i, name)



# %%
from torch_geometric.data import HeteroData

data = HeteroData()
# %%
import torch

data['ticker'].x = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float)
data['ticker'].name = ['AAPL', 'GOOGL', 'AMZN']
# %%
data

# %%
from torch_geometric.utils import to_networkx

# %%
keu = to_networkx(data, node_attrs=['name'])
# %%
keu
# %%
dir(keu.nodes)
# %%
keu.nodes[0]
# %%
keu.nodes.data()
# %%
import networkx as nx
G = nx.Graph()

G.add_nodes_from([(0, {"color": "red", "name":"APPL"}), (5, {"color": "green", "":"GOOGL"}), (10, {"color": "blue", "name":"AMZN"})])
# %%
G.nodes[0]
# %% 
import pandas as pd

predicted_data = pd.read_csv("financial_kg_ds/data/predictions/prediction_2024_01_14.csv")
# %%
predicted_data
# %%
