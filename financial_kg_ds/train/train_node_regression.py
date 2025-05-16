# %%
import os
import json
from torch_geometric.transforms import ToUndirected
from financial_kg_ds.datasets.graph_loader import GraphLoaderRegresion
from financial_kg_ds.models.GNN_hetero_sage_conv import HeteroGNN

# %%
data = GraphLoaderRegresion.get_data()
data = ToUndirected()(data)

# %%
import torch
import torch.nn.functional as F
from torch.nn import Linear
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
        hidden_channels=trial.suggest_int("hidden_channels", 32, 256, log=True),
        out_channels=1,
        num_layers=trial.suggest_int("num_layers", 2, 5),
        dropout=trial.suggest_float("dropout", 0.1, 0.5),
        gnn_aggr=trial.suggest_categorical("gnn_aggr", ["add", "mean", "max"]),
    )


def save_checkpoint(model, trial_number, val_loss, params, checkpoint_dir="checkpoints"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'val_loss': val_loss,
        'params': params
    }
    path = os.path.join(checkpoint_dir, f"model_trial_{trial_number}.pt")
    torch.save(checkpoint, path)

def validate(model, data):
    model.eval()
    with torch.no_grad():
        out = model(data.x_dict, data.edge_index_dict)
        mask = data["ticker"].val_mask
        val_loss = F.mse_loss(out[mask], data["ticker"].y[mask]).item()
    return val_loss

def objective(trial):
    val_loss_min = float('inf')
    patience = 5
    patience_counter = 0
    
    model = define_model(trial)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=3,
        verbose=True
    )
    
    num_epochs = trial.suggest_int("num_epochs", 20, 100)  # Make epochs configurable
    
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x_dict, data.edge_index_dict)
        mask = data["ticker"].train_mask
        loss = F.mse_loss(out[mask], data["ticker"].y[mask])
        loss.backward()
        optimizer.step()

        # Validation
        val_loss = validate(model, data)
        scheduler.step(val_loss)
            
        if val_loss < val_loss_min:
            val_loss_min = val_loss
            patience_counter = 0
            print(f"Epoch {epoch}: Validation Loss decreased to {val_loss_min}")
            trial.set_user_attr("best_model", model.state_dict())
            save_checkpoint(model, trial.number, val_loss_min, trial.params)  # Save checkpoint
        else:
            patience_counter += 1
                
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break
                
    return val_loss_min

# %%
study = optuna.create_study(
    direction="minimize",
    pruner=optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=10
    )
)
study.optimize(objective, n_trials=50)  # Increase number of trials

# %%
study.best_params

# %%
plot_param_importances(study)
optuna.visualization.plot_intermediate_values(study).show()
fig = optuna.visualization.plot_contour(study, params=["hidden_channels", "gnn_aggr"])
fig.show()


# %% Load the best model
best_model = define_model(study.best_trial)
best_model.load_state_dict(study.best_trial.user_attrs["best_model"])

# %% load new data
data_new = GraphLoaderRegresion("C:/Users/Admin/Desktop/FINANCIAL_KG/data/data_2024-01-14").get_data()
data_new = ToUndirected()(data_new)

# %% Evaluate the model
best_model.eval()
out = best_model(data_new.x_dict, data_new.edge_index_dict)
mask = data_new["ticker"].test_mask
loss = F.mse_loss(out[mask], data_new["ticker"].y[mask])
print(f"Test Loss: {loss.item()}")

# %% check the predictions
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error

def plot_predictions(y_true, y_pred):
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('True vs Predicted Values')
    plt.show()

def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R^2: {r2:.4f}")

    return mse, rmse, mae, r2


# %%
out.shape

# %%
pred_val = data["ticker"].y
# %%
tickers = data_new["ticker"].name
# %%
pred_df = pd.DataFrame([tickers, pred_val.squeeze(1).detach().numpy()]).T
# %%
pred_df.to_csv("financial_kg_ds/data/predictions/prediction_2025_01_14.csv")

# %%
