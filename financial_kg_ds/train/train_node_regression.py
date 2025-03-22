# %%
from torch_geometric.transforms import ToUndirected
from financial_kg_ds.datasets.graph_loader import GraphLoaderRegresion

data = GraphLoaderRegresion().get_data()
data = ToUndirected()(data)

# %%
for i in data["institution"]["x"][0]:
    print(i)

# %%
import torch
import torch.nn.functional as F
from torch.nn import Linear

# %%
from torch_geometric.nn import GATConv, HeteroConv, SAGEConv


class HeteroGNN(torch.nn.Module):
    def __init__(self, metadata, hidden_channels, out_channels, num_layers, gnn_aggr="add"):
        super().__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv(
                {
                    edge_type: SAGEConv((-1, -1), hidden_channels, aggr=gnn_aggr)
                    # edge_type: GATConv((-1, -1), hidden_channels, dropout=0.7, add_self_loops=False)
                    for edge_type in metadata[1]
                }
            )
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.dropout(x, p=0.2, training=self.training) for key, x in x_dict.items()}
            x_dict = {key: F.leaky_relu(x) for key, x in x_dict.items()}
        return self.lin(x_dict["ticker"])

import optuna
from optuna.visualization import plot_param_importances

# %%
model = HeteroGNN(data.metadata(), hidden_channels=16, out_channels=1, num_layers=2)
out = model(data.x_dict, data.edge_index_dict)
mask = data["ticker"].train_mask
loss = F.mse_loss(out[mask], data["ticker"].y[mask])

# %%
def define_model(trial):
    return HeteroGNN(
        data.metadata(),
        hidden_channels=trial.suggest_int("hidden_channels", 16, 64),
        out_channels=1,
        num_layers=trial.suggest_int("num_layers", 1, 3),
        gnn_aggr=trial.suggest_categorical("gnn_aggr", ["add", "mean", "max"]),
    )

def objective(trial):

    val_loss_min = float('inf')  # initialize minimum validation loss

    model = define_model(trial)# .to(data.metadata().device)

    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for _ in range(1, 20):
        model.train()
        optimizer.zero_grad()
        out = model(data.x_dict, data.edge_index_dict)
        mask = data["ticker"].train_mask
        loss = F.mse_loss(out[mask], data["ticker"].y[mask])
        loss.backward()
        optimizer.step()

    model.eval()
    out = model(data.x_dict, data.edge_index_dict)

    mask = data["ticker"].val_mask
    val_loss = F.mse_loss(out[mask], data["ticker"].y[mask]).item()
    if val_loss < val_loss_min:
        val_loss_min = val_loss
        print(f"Validation Loss decreased to {val_loss_min}")

    return val_loss

# %%
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=10)

# %%
study.best_params

# %%
plot_param_importances(study)
optuna.visualization.plot_intermediate_values(study).show()
fig = optuna.visualization.plot_contour(study, params=["hidden_channels", "gnn_aggr"])
fig.show()

