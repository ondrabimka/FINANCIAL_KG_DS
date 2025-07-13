# %%
import os
import yaml
import torch
import torch.nn.functional as F
from torch_geometric.transforms import ToUndirected
from financial_kg_ds.datasets.graph_loader import GraphLoaderRegresion
from financial_kg_ds.models.GNN_hetero_sage_conv import HeteroGNN
from financial_kg_ds.utils.mlflow_utils import MLflowTracker
from financial_kg_ds.utils.evaluate_gnn import ModelEvaluator
from financial_kg_ds.utils.losses import LossFactory
from datetime import datetime
import mlflow
import optuna
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Config loading ---
def load_yaml_config(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)

MODEL_CONFIG_PATH = "configs/models/base_gnn.yaml"
TRAIN_CONFIG_PATH = "configs/training/default_training.yaml"

model_config = load_yaml_config(MODEL_CONFIG_PATH)
train_config = load_yaml_config(TRAIN_CONFIG_PATH)

# --- Data loading ---
data = GraphLoaderRegresion.get_data()
data = ToUndirected()(data)

# --- Model definition ---
def define_model(trial):
    params = model_config['model']['optuna_params']
    return HeteroGNN(
        data.metadata(),
        hidden_channels=trial.suggest_int(
            "hidden_channels", params['hidden_channels']['min'], params['hidden_channels']['max'], log=params['hidden_channels'].get('log', False)
        ),
        out_channels=model_config['model']['fixed_params']['out_channels'],
        num_layers=trial.suggest_int(
            "num_layers", params['num_layers']['min'], params['num_layers']['max']
        ),
        gnn_aggr=trial.suggest_categorical(
            "gnn_aggr", params['gnn_aggr']['choices']
        ),
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
    model.train()
    optimizer.zero_grad()
    out = model(data.x_dict, data.edge_index_dict)
    mask = data["ticker"].train_mask
    loss = loss_fn(out[mask], data["ticker"].y[mask])
    loss.backward()
    optimizer.step()
    return loss.item()

def validate(model, data, loss_fn):
    model.eval()
    with torch.no_grad():
        out = model(data.x_dict, data.edge_index_dict)
        mask = data["ticker"].val_mask
        val_loss = loss_fn(out[mask], data["ticker"].y[mask])
    return val_loss

def objective(trial):
    mlflow_tracker = MLflowTracker("GNN_Optimization")
    run_name = f"trial_{trial.number}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    mlflow_tracker.start_run(run_name, nested=True)

    try:
        loss_fn = LossFactory.create_loss(model_config)
        patience = train_config['training']['early_stopping']['patience']
        num_epochs = train_config['training']['num_epochs']
        val_loss_min = float('inf')
        patience_counter = 0

        model = define_model(trial)
        lr_params = train_config['training']['optuna']['learning_rate']
        learning_rate = trial.suggest_float(
            "learning_rate", lr_params['min'], lr_params['max'], log=lr_params.get('log', False)
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        mlflow_tracker.log_params(trial.params)

        for epoch in range(num_epochs):
            train_loss = train(model, data, optimizer, loss_fn)
            val_loss = validate(model, data, loss_fn)
            mlflow_tracker.log_metrics({"train_loss": train_loss, "val_loss": val_loss}, step=epoch)

            if val_loss < val_loss_min:
                val_loss_min = val_loss
                patience_counter = 0
                trial.set_user_attr("best_model", model.state_dict())
            else:
                patience_counter += 1

            save_checkpoint(model, trial.number, val_loss, trial.params)
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            if patience_counter >= patience:
                print("Early stopping triggered")
                break

        mlflow_tracker.log_model(model, "model", data)
        mlflow_tracker.log_metrics({"final_val_loss": val_loss_min, "best_epoch": epoch})
        return val_loss_min

    finally:
        mlflow_tracker.end_run()

def main():
    if mlflow.active_run():
        mlflow.end_run()

    mlflow_tracker = MLflowTracker("GNN_Training")
    run_name = f"full_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    mlflow_tracker.start_run(run_name)

    try:
        data_path = os.getenv("TRAIN_DATA_PATH")
        eval_data_path = os.getenv("EVAL_DATA_PATH")
        test_data_path = os.getenv("TEST_DATA_PATH")
        for path in [data_path, eval_data_path, test_data_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Data path not found: {path}")

        mlflow_tracker.log_params(model_config)
        mlflow_tracker.log_params(train_config)

        optuna_trials = train_config['training']['optuna']['n_trials']
        study = optuna.create_study(direction=train_config['training']['optuna']['direction'])
        study.optimize(objective, n_trials=optuna_trials)

        best_model = define_model(study.best_trial)
        best_model.load_state_dict(study.best_trial.user_attrs["best_model"])

        data_new = GraphLoaderRegresion(
            data_path=eval_data_path,
            eval_data_path=test_data_path
        ).get_data()
        data_new = ToUndirected()(data_new)

        eval_cfg = train_config['evaluation']
        evaluator = ModelEvaluator(
            eval_data_path, test_data_path,
            threshold=eval_cfg['threshold'],
            prediction_limit=eval_cfg['prediction_limit']
        )
        metrics, eval_df, eval_plots = evaluator.evaluate(best_model, data_new)
        save_dir = "financial_kg_ds/experiments/evaluations"
        result_paths = evaluator.save_results(eval_df, metrics, eval_plots, save_dir)

        mlflow_tracker.log_metrics(metrics)
        for path_type, path in result_paths.items():
            mlflow_tracker.log_artifact(path)

        model_name = f"best_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        mlflow_tracker.log_model(model=best_model, name=model_name, data=data_new)

        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, (int, float)):
                mlflow_tracker.log_metrics({f"final_{metric_name}": float(metric_value)})

    finally:
        mlflow_tracker.end_run()

if __name__ == "__main__":
    main()

