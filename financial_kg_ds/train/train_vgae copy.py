# %%
from financial_kg_ds.datasets.graph_loader import GraphLoader
from financial_kg_ds.models.GVAE import VariationalGraphAutoEncoder
from torch_geometric.transforms import ToUndirected
from torch_geometric.nn import to_hetero

# data = GraphLoader().get_data()
# data = ToUndirected()(data)

from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform = T.Compose([
    T.NormalizeFeatures(),
    T.ToDevice(device),
    T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=True,
                      split_labels=True, add_negative_train_samples=False),
])

dataset = Planetoid('data/', 'Cora', transform=transform)
train_data, val_data, test_data = dataset[0]


# %%
from torch_geometric.nn import GCNConv, VGAE

class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels)
        self.conv_mu = GCNConv(2 * out_channels, out_channels)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)
    
in_channels, out_channels = dataset.num_features, 16
model = VGAE(VariationalGCNEncoder(in_channels, out_channels))


# %%
print(train_data.pos_edge_label)

# %%
import time

model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def train():
    model.train()
    optimizer.zero_grad()
    z = model.encode(train_data.x, train_data.edge_index)
    loss = model.recon_loss(z, train_data.pos_edge_label_index)
    loss = loss + (1 / train_data.num_nodes) * model.kl_loss()
    loss.backward()
    optimizer.step()
    return float(loss)

@torch.no_grad()
def test(data):
    model.eval()
    z = model.encode(data.x, data.edge_index)
    return model.test(z, data.pos_edge_label_index, data.neg_edge_label_index)

# %%
times = []
for epoch in range(1, 1001):
    start = time.time()
    loss = train()
    auc, ap = test(test_data)
    print(f'Epoch: {epoch:03d}, AUC: {auc:.4f}, AP: {ap:.4f}')
    times.append(time.time() - start)
print(f"Median time per epoch: {torch.tensor(times).median():.4f}s")

# %%
