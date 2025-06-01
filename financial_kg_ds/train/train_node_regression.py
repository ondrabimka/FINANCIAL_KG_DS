# %%
import os
import json
from torch_geometric.transforms import ToUndirected
from financial_kg_ds.datasets.graph_loader import GraphLoaderRegresion
from financial_kg_ds.models.GNN_hetero_sage_conv import HeteroGNN
import torch
import numpy as np
import pandas as pd
from financial_kg_ds.utils.mlflow_utils import MLflowTracker
from datetime import datetime
import yaml
import joblib
from financial_kg_ds.utils.evaluate_gnn import ModelEvaluator
from financial_kg_ds.utils.losses import LossFactory
import mlflow

# %% Hyperparameters
NUM_EPOCHS = 100  # Number of epochs for training

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


def train(model, data, optimizer, loss_fn):
    """ Train model for one epoch """
    model.train()
    optimizer.zero_grad()
    
    out = model(data.x_dict, data.edge_index_dict)
    mask = data["ticker"].train_mask
    loss = loss_fn(out[mask], data["ticker"].y[mask])
    
    loss.backward()
    optimizer.step()
    
    return loss.item()

def validate(model, data, loss_fn):
    """Validate model """
    model.eval()
    with torch.no_grad():
        out = model(data.x_dict, data.edge_index_dict)
        mask = data["ticker"].val_mask
        val_loss = loss_fn(out[mask], data["ticker"].y[mask])
    return val_loss


def load_config():
    with open("configs/models/base_gnn.yaml", "r") as f:
        return yaml.safe_load(f)

def objective(trial):
    # Initialize MLflow tracking with nested=True
    mlflow_tracker = MLflowTracker("GNN_Optimization")
    run_name = f"trial_{trial.number}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    mlflow_tracker.start_run(run_name, nested=True)  # Add nested=True here

    try:

        config = load_config()
        loss_fn = LossFactory.create_loss(config)

        val_loss_min = float('inf')
        patience = 5
        patience_counter = 0
        
        model = define_model(trial)
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        # Log parameters
        mlflow_tracker.log_params({
            "hidden_channels": trial.params["hidden_channels"],
            "num_layers": trial.params["num_layers"],
            "gnn_aggr": trial.params["gnn_aggr"],
            "learning_rate": trial.params["learning_rate"]
        })
        
        for epoch in range(NUM_EPOCHS):
            train_loss = train(model, data, optimizer, loss_fn)
            val_loss = validate(model, data, loss_fn)
            
            # Log metrics
            mlflow_tracker.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss
            }, step=epoch)
            
            # Save the best model
            if val_loss < val_loss_min:
                val_loss_min = val_loss
                patience_counter = 0
                trial.set_user_attr("best_model", model.state_dict())
            else:
                patience_counter += 1
            
            # Save checkpoint
            save_checkpoint(model, trial.number, val_loss, trial.params)
            
            print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            if patience_counter >= patience:
                print("Early stopping triggered")
                break
            
        # Log final model and metrics
        mlflow_tracker.log_model(model, "model")
        mlflow_tracker.log_metrics({
            "final_val_loss": val_loss_min,
            "best_epoch": epoch
        })
        
        return val_loss_min
    
    finally:
        mlflow_tracker.end_run()

def main():
    # End any existing runs (safety check)
    if mlflow.active_run():
        mlflow.end_run()

    try:
        config = load_config()
    except FileNotFoundError:
        print("Error: Config file not found. Please ensure base_gnn.yaml exists in configs/models/")
        return

    # Create MLflow experiment for the full training
    mlflow_tracker = MLflowTracker("GNN_Training")
    run_name = f"full_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    mlflow_tracker.start_run(run_name)  # This will be the parent run
    
    try:
        # Ensure data path exists
        data_path = "C:/Users/Admin/Desktop/FINANCIAL_KG/data/data_2024-11-15"
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data path not found: {data_path}")
            
        # Log configuration
        mlflow_tracker.log_params(config)
        
        # Run optimization
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=2)
        
        # Train final model with best parameters
        best_model = define_model(study.best_trial)
        best_model.load_state_dict(study.best_trial.user_attrs["best_model"])
        
        # Evaluate on new data
        data_new = GraphLoaderRegresion("C:/Users/Admin/Desktop/FINANCIAL_KG/data/data_2024-11-15").get_data()
        data_new = ToUndirected()(data_new)

        # Initialize evaluator
        evaluator = ModelEvaluator(threshold=100, prediction_limit=5)
        
        # Evaluate model
        metrics, eval_df, eval_plots = evaluator.evaluate(best_model, data_new)
        
        # Save evaluation results
        save_dir = "financial_kg_ds/experiments/evaluations"
        result_paths = evaluator.save_results(eval_df, metrics, eval_plots, save_dir)
        
        # Log to MLflow
        mlflow_tracker.log_metrics(metrics)
        for path_type, path in result_paths.items():
            mlflow_tracker.log_artifact(path)
        
    finally:
        mlflow_tracker.end_run()

if __name__ == "__main__":
    main()