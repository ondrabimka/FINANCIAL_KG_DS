# %%
from financial_kg_ds.datasets.graph_loader import GraphLoaderRegresion
from torch_geometric.transforms import ToUndirected

from torch_geometric.datasets import DBLP

data = GraphLoaderRegresion().get_data()
data = ToUndirected()(data)

# %%
for i in data['institution']['x'][0]:
    print(i)

# %%
data.metadata()[1]


# %%
from torch_geometric.nn import SAGEConv, HeteroConv, GATConv
import torch
from torch.nn import Linear
import torch.nn.functional as F

class HeteroGNN(torch.nn.Module):
    def __init__(self, metadata, hidden_channels, out_channels, num_layers):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                edge_type: SAGEConv((-1, -1), hidden_channels)
                # edge_type: GATConv((-1, -1), hidden_channels, dropout=0.7, add_self_loops=False)
                for edge_type in metadata[1]
            })
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        for conv in self.convs: 
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.dropout(x, p=0.7, training=self.training) for key, x in x_dict.items()}
            x_dict = {key: F.leaky_relu(x) for key, x in x_dict.items()}
        return self.lin(x_dict['ticker'])


# %%
model = HeteroGNN(data.metadata(), hidden_channels=16, out_channels=1,
                  num_layers=2)

# %%
out = model(data.x_dict, data.edge_index_dict)
# %%
out
# %%
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x_dict, data.edge_index_dict)
    mask = data['ticker'].train_mask
    loss = F.mse_loss(out[mask], data['ticker'].y[mask])
    loss.backward()
    optimizer.step()
    return loss.item()

# %%
@torch.no_grad()
def test():
    model.eval()
    out = model(data.x_dict, data.edge_index_dict)

    accs = []
    for key in ['train_mask', 'val_mask', 'test_mask']:
        mask = data['ticker'][key]
        acc = F.mse_loss(out[mask], data['ticker'].y[mask])
        accs.append(acc)
    return accs

# %% test
for epoch in range(1, 101):
    loss = train()
    train_acc, val_acc, test_acc = test()
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, '
          f'Val: {val_acc:.4f}, Test: {test_acc:.4f}')

# Epoch: 027, Loss: 1.5697, Train: 0.8947, Val: 0.9431, Test: 1.1420 GraphConv
# Epoch: 054, Loss: 0.3511, Train: 0.1847, Val: 0.9222, Test: 1.0981

# %% check validation
out = model(data.x_dict, data.edge_index_dict)
# %%
out
# %%
