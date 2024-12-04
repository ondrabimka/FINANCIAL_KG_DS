# %%
from typing import Optional

import torch
from torch import Tensor
from torch_geometric.nn import to_hetero
from torch_geometric.transforms import ToUndirected
from torch_geometric.utils import negative_sampling

from financial_kg_ds.datasets.graph_loader import GraphLoaderAE
from financial_kg_ds.models.GAE import SimpleHeteroGAE

device = "cuda" if torch.cuda.is_available() else "cpu"

data = GraphLoaderAE().get_data()
data = ToUndirected()(data)
# del data[('mutual_fund', 'rev_holds_mt', 'ticker')]
# del data[('institution', 'rev_holds_it', 'ticker')]

# %%
data_loader = GraphLoaderAE()
data_loader.load_full_graph()

# %%
from torch.nn import functional as F

hidden_channels = 16
model = SimpleHeteroGAE(hidden_channels).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.025)

# %%
def train():
    model.train()
    optimizer.zero_grad()

    # Get the embeddings (output) from the model
    output = model(data.x_dict, data.edge_index_dict)

    total_loss = 0
    for node_type in data.node_types:
        # Get embeddings for the current node type
        node_out = output[node_type]

        # Get the target adjacency matrix for the current node type
        target_matrix = data[node_type].x

        # Calculate the loss using MSE (reconstructing identity or adjacency matrix)
        loss = F.mse_loss(node_out, target_matrix)

        # Accumulate the loss for this node type
        total_loss += loss

    # Backpropagate and optimize
    total_loss.backward()
    optimizer.step()

    return total_loss.item()


for epoch in range(1, 20):
    loss = train()
    print(f"Epoch {epoch:03d}, Loss: {loss:.4f}")

# %%
ticker_embeddings = model.encode(data.x_dict, data.edge_index_dict)["ticker"]


# %% convert the embeddings to numpy
import pandas as pd

ticker_mapping = data_loader.ticker_mapping

# %%
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
z_ticker_2d = pca.fit_transform(ticker_embeddings.detach().numpy())
dataframe = pd.DataFrame(list(zip(ticker_mapping.keys(), z_ticker_2d)), columns=["ticker", "reduced_embeddings"])
dataframe["x"] = dataframe["reduced_embeddings"].apply(lambda x: x[0])
dataframe["y"] = dataframe["reduced_embeddings"].apply(lambda x: x[1])

# %% plot the reduced embeddings using plotly
import plotly.express as px

fig = px.scatter(dataframe, x="x", y="y", text="ticker")
fig.show()

# %%
ticker_df = pd.read_csv(
    "/Users/obimka/Desktop/Zabafa/FINANCIAL_KG/data/data_2024-09-06/ticker_info.csv", usecols=["ticker", "industry", "marketCap"]
)
# %%
ticker_df = ticker_df.merge(dataframe, on="ticker")
# %%
ticker_df

# %% plot the reduced embeddings using plotly
import plotly.express as px

fig = px.scatter(ticker_df.dropna(), x="x", y="y", color="industry", hover_data=["ticker", "marketCap"])
fig.show()
# %%
