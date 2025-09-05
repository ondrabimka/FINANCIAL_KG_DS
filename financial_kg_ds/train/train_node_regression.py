# %%
import os
import yaml
import torch
import torch.nn.functional as F
import numpy as np
import argparse
import sys
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
import pandas as pd

# Load environment variables
load_dotenv()

def parse_arguments():
    """Parse command line arguments for training configuration."""
    parser = argparse.ArgumentParser(description="Train Financial GNN Model")
    
    # Model configuration
    parser.add_argument("--hidden-channels", type=int, default=None,
                      help="Number of hidden channels (overrides config)")
    parser.add_argument("--num-layers", type=int, default=None,
                      help="Number of GNN layers (overrides config)")
    parser.add_argument("--dropout", type=float, default=None,
                      help="Dropout rate (overrides config)")
    
    # Training configuration
    parser.add_argument("--epochs", type=int, default=None,
                      help="Number of training epochs (overrides config)")
    parser.add_argument("--trials", type=int, default=None,
                      help="Number of Optuna trials (overrides config)")
    parser.add_argument("--learning-rate", type=float, default=None,
                      help="Learning rate (overrides config)")
    
    # Loss function
    parser.add_argument("--loss", type=str, default=None, 
                      choices=["mse", "asymmetric", "huber", "quantile"],
                      help="Loss function to use (overrides config)")
    
    # Training modes
    parser.add_argument("--quick", action="store_true",
                      help="Quick training mode (fewer epochs and trials)")
    parser.add_argument("--no-eval", action="store_true",
                      help="Skip evaluation stages")
    
    # Configuration files
    parser.add_argument("--model-config", type=str, default=None,
                      help="Path to model configuration file")
    parser.add_argument("--train-config", type=str, default=None,
                      help="Path to training configuration file")
    
    # Data paths
    parser.add_argument("--train-data", type=str, default=None,
                      help="Path to training data (overrides env var)")
    parser.add_argument("--eval-data", type=str, default=None,
                      help="Path to evaluation data (overrides env var)")
    parser.add_argument("--test-data", type=str, default=None,
                      help="Path to test data (overrides env var)")
    
    # Output
    parser.add_argument("--output-dir", type=str, default=None,
                      help="Directory to save models and results")
    parser.add_argument("--experiment-name", type=str, default="GNN_Financial_Training",
                      help="MLflow experiment name")
    
    return parser.parse_args()

def load_and_override_configs(args):
    """Load YAML configs and override with command line arguments."""
    # Determine paths
    if hasattr(args, 'model_config') and args.model_config:
        model_config_path = args.model_config
    else:
        model_config_path = os.path.join(os.getcwd(), "configs/models/base_gnn.yaml")
    
    if hasattr(args, 'train_config') and args.train_config:
        train_config_path = args.train_config  
    else:
        train_config_path = os.path.join(os.getcwd(), "configs/training/default_training.yaml")
    
    # Load configurations
    model_config = load_yaml_config(model_config_path)
    train_config = load_yaml_config(train_config_path)
    
    # Override model config with command line args
    if args.hidden_channels is not None:
        model_config.setdefault('model', {}).setdefault('fixed_params', {})['hidden_channels'] = args.hidden_channels
    if args.num_layers is not None:
        model_config.setdefault('model', {}).setdefault('fixed_params', {})['num_layers'] = args.num_layers
    if args.dropout is not None:
        model_config.setdefault('model', {}).setdefault('fixed_params', {})['dropout'] = args.dropout
    if args.loss is not None:
        model_config.setdefault('loss', {})['name'] = args.loss
    
    # Override training config with command line args
    if args.epochs is not None:
        train_config.setdefault('training', {})['num_epochs'] = args.epochs
    if args.trials is not None:
        train_config.setdefault('training', {}).setdefault('optuna', {})['n_trials'] = args.trials
    if args.learning_rate is not None:
        train_config.setdefault('training', {}).setdefault('optuna', {}).setdefault('learning_rate', {})['min'] = args.learning_rate
        train_config['training']['optuna']['learning_rate']['max'] = args.learning_rate
    
    # Quick mode adjustments
    if args.quick:
        train_config.setdefault('training', {})['num_epochs'] = min(train_config['training'].get('num_epochs', 50), 10)
        train_config.setdefault('training', {}).setdefault('optuna', {})['n_trials'] = min(train_config['training']['optuna'].get('n_trials', 10), 3)
        print("🏃 Quick mode enabled - reduced epochs and trials")
    
    return model_config, train_config

# --- Config loading ---
def load_yaml_config(path):
    """Load YAML configuration with proper error handling."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    
    try:
        with open(path, "r") as f:
            config = yaml.safe_load(f)
        print(f"✓ Loaded config from: {path}")
        return config
    except yaml.YAMLError as e:
        raise ValueError(f"Error parsing YAML file {path}: {e}")

# Use relative paths for config files from project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "models", "base_gnn.yaml")
TRAIN_CONFIG_PATH = os.path.join(PROJECT_ROOT, "configs", "training", "default_training.yaml")

print("Loading configuration files...")
model_config = load_yaml_config(MODEL_CONFIG_PATH)
train_config = load_yaml_config(TRAIN_CONFIG_PATH)

print("Configuration loaded successfully:")
print(f"Model: {model_config['model']['name']}")
print(f"Loss function: {model_config['loss']['name']}")
print(f"Training epochs: {train_config['training']['num_epochs']}")
print(f"Optuna trials: {train_config['training']['optuna']['n_trials']}")

# --- Data loading ---
data = GraphLoaderRegresion.get_data()
data = ToUndirected()(data)

# --- Model definition ---
def define_model(trial):
    """Define model with trial parameters and enhanced architecture"""
    params = model_config['model']['optuna_params']
    fixed_params = model_config['model']['fixed_params']
    
    return HeteroGNN(
        data.metadata(),
        hidden_channels=trial.suggest_int(
            "hidden_channels", 
            params['hidden_channels']['min'], 
            params['hidden_channels']['max'], 
            log=params['hidden_channels'].get('log', False)
        ),
        out_channels=fixed_params['out_channels'],
        num_layers=trial.suggest_int(
            "num_layers", 
            params['num_layers']['min'], 
            params['num_layers']['max']
        ),
        gnn_aggr=trial.suggest_categorical(
            "gnn_aggr", 
            params['gnn_aggr']['choices']
        ),
        dropout=fixed_params.get('dropout', 0.2)
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
    """Validation function with additional metrics tracking"""
    model.eval()
    with torch.no_grad():
        out = model(data.x_dict, data.edge_index_dict)
        mask = data["ticker"].val_mask
        val_loss = loss_fn(out[mask], data["ticker"].y[mask])
        
        # Additional validation metrics for financial evaluation
        predictions = out[mask].cpu().numpy()
        targets = data["ticker"].y[mask].cpu().numpy()
        
        # Direction accuracy - key metric for trading
        pred_direction = np.sign(predictions)
        true_direction = np.sign(targets)
        direction_accuracy = np.mean(pred_direction == true_direction)
        
        # Mean absolute error
        mae = np.mean(np.abs(predictions - targets))
        
    return val_loss, direction_accuracy, mae

def calculate_financial_metrics(predictions, actuals, returns_df=None):
    """Calculate comprehensive financial performance metrics"""
    metrics = {}
    
    # Basic regression metrics
    mse = np.mean((predictions - actuals) ** 2)
    metrics['mse'] = float(mse)
    metrics['rmse'] = float(np.sqrt(mse))
    metrics['mae'] = float(np.mean(np.abs(predictions - actuals)))
    
    # Direction-based metrics
    pred_direction = np.sign(predictions)
    true_direction = np.sign(actuals)
    direction_accuracy = np.mean(pred_direction == true_direction)
    metrics['direction_accuracy'] = float(direction_accuracy)
    
    # Financial performance metrics
    if returns_df is not None:
        # Trading simulation based on predictions
        trades = returns_df.copy()
        trades['signal'] = np.where(predictions > 0, 1, -1)  # Buy/sell signal
        trades['strategy_return'] = trades['signal'] * trades['actual_return']
        
        # Portfolio metrics
        total_return = trades['strategy_return'].sum()
        win_rate = (trades['strategy_return'] > 0).mean()
        avg_win = trades[trades['strategy_return'] > 0]['strategy_return'].mean() if (trades['strategy_return'] > 0).any() else 0
        avg_loss = trades[trades['strategy_return'] < 0]['strategy_return'].mean() if (trades['strategy_return'] < 0).any() else 0
        
        metrics.update({
            'total_return_pct': float(total_return),
            'win_rate': float(win_rate),
            'avg_win_pct': float(avg_win),
            'avg_loss_pct': float(avg_loss),
            'profit_factor': float(avg_win / abs(avg_loss)) if avg_loss != 0 else float('inf'),
            'num_trades': int(len(trades)),
            'sharpe_ratio': float(trades['strategy_return'].mean() / trades['strategy_return'].std()) if trades['strategy_return'].std() != 0 else 0
        })
    
    return metrics

# Global variable to store the best model across all trials
best_global_model_state = None
best_global_loss = float('inf')

def objective(trial):
    """Optuna objective function with enhanced financial metrics tracking"""
    global best_global_model_state, best_global_loss
    
    mlflow_tracker = MLflowTracker("GNN_Optimization")
    run_name = f"trial_{trial.number}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    mlflow_tracker.start_run(run_name, nested=True)

    try:
        loss_fn = LossFactory.create_loss(model_config)
        patience = train_config['training']['early_stopping']['patience']
        num_epochs = train_config['training']['num_epochs']
        val_loss_min = float('inf')
        best_direction_accuracy = 0.0
        patience_counter = 0

        model = define_model(trial)
        lr_params = train_config['training']['optuna']['learning_rate']
        learning_rate = trial.suggest_float(
            "learning_rate", lr_params['min'], lr_params['max'], log=lr_params.get('log', False)
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Log trial parameters
        trial_params = trial.params.copy()
        trial_params['learning_rate'] = learning_rate
        mlflow_tracker.log_params(trial_params)

        for epoch in range(num_epochs):
            train_loss = train(model, data, optimizer, loss_fn)
            val_loss, direction_accuracy, mae = validate(model, data, loss_fn)
            
            # Log metrics for this epoch
            epoch_metrics = {
                "train_loss": train_loss,
                "val_loss": val_loss.item() if hasattr(val_loss, 'item') else val_loss,
                "direction_accuracy": direction_accuracy,
                "val_mae": mae
            }
            mlflow_tracker.log_metrics(epoch_metrics, step=epoch)

            if val_loss < val_loss_min:
                val_loss_min = val_loss
                best_direction_accuracy = direction_accuracy
                patience_counter = 0
                # Store the best model state dict in a way that can be retrieved
                trial.set_user_attr("best_model_state", model.state_dict())
                trial.set_user_attr("best_epoch", epoch)
                
                # Also save globally if this is the best across all trials
                if val_loss < best_global_loss:
                    best_global_loss = val_loss
                    best_global_model_state = model.state_dict().copy()
                    
            else:
                patience_counter += 1

            save_checkpoint(model, trial.number, val_loss, trial_params)
            
            if epoch % 10 == 0:  # Print every 10 epochs
                print(f"Trial {trial.number} - Epoch {epoch+1}/{num_epochs}, "
                      f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                      f"Direction Acc: {direction_accuracy:.3f}")

            if patience_counter >= patience:
                print(f"Trial {trial.number} - Early stopping triggered at epoch {epoch}")
                break

        # Log final trial metrics
        final_metrics = {
            "final_val_loss": float(val_loss_min),
            "final_direction_accuracy": float(best_direction_accuracy),
            "best_epoch": trial.user_attrs.get("best_epoch", epoch),
            "total_epochs": epoch + 1
        }
        mlflow_tracker.log_metrics(final_metrics)
        
        return float(val_loss_min)

    except Exception as e:
        print(f"Trial {trial.number} failed with error: {e}")
        mlflow_tracker.log_metrics({"trial_failed": 1.0})
        return float('inf')
    finally:
        mlflow_tracker.end_run()

def evaluate_financial_performance(model, data_path_start, data_path_end, stage_name=""):
    """Comprehensive financial evaluation on specific data period"""
    
    # Load data for evaluation
    eval_data = GraphLoaderRegresion(data_path=data_path_start).get_data()
    eval_data = ToUndirected()(eval_data)
    
    # Get model predictions
    model.eval()
    with torch.no_grad():
        predictions = model(eval_data.x_dict, eval_data.edge_index_dict).cpu().numpy().squeeze()
    
    # Use ModelEvaluator for comprehensive financial metrics
    evaluator = ModelEvaluator(
        data_path_start, data_path_end,
        threshold=train_config['evaluation']['threshold'],
        prediction_limit=train_config['evaluation']['prediction_limit']
    )
    
    try:
        financial_metrics, eval_df, eval_plots = evaluator.evaluate(model, eval_data)
        
        # Add stage prefix to metrics
        stage_metrics = {f"{stage_name}_{k}": v for k, v in financial_metrics.items()}
        
        return stage_metrics, eval_df, eval_plots
        
    except Exception as e:
        print(f"Warning: Financial evaluation failed for {stage_name}: {e}")
        # Return basic metrics as fallback
        return {
            f"{stage_name}_prediction_count": len(predictions),
            f"{stage_name}_prediction_mean": float(np.mean(predictions)),
            f"{stage_name}_prediction_std": float(np.std(predictions))
        }, None, None

def main():
    """Enhanced main function with three-stage evaluation pipeline"""
    global model_config, train_config, best_global_model_state, best_global_loss
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Load and override configurations
    global_model_config, global_train_config = load_and_override_configs(args)
    model_config.update(global_model_config)
    train_config.update(global_train_config)
    
    # Reset global variables
    best_global_model_state = None
    best_global_loss = float('inf')
    
    if mlflow.active_run():
        mlflow.end_run()

    mlflow_tracker = MLflowTracker("GNN_Financial_Training")
    run_name = f"financial_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    mlflow_tracker.start_run(run_name)

    try:
        # Use CLI arguments for data paths or fallback to environment variables
        train_data_path = args.train_data or os.getenv("TRAIN_DATA_PATH")
        eval_data_path = args.eval_data or os.getenv("EVAL_DATA_PATH") 
        test_data_path = args.test_data or os.getenv("TEST_DATA_PATH")
        
        data_paths = {
            "train": train_data_path,
            "eval": eval_data_path, 
            "test": test_data_path
        }
        
        print("\n=== Data Path Verification ===")
        for stage, path in data_paths.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"{stage.upper()} data path not found: {path}")
            print(f"✓ {stage.upper()}: {path}")

        # Log configuration
        mlflow_tracker.log_params({
            "model_config": model_config,
            "train_config": train_config,
            "train_data_path": train_data_path,
            "eval_data_path": eval_data_path,
            "test_data_path": test_data_path
        })

        print("\n=== Stage 1: Hyperparameter Optimization on Training Data ===")
        # Use CLI args or config defaults
        optuna_trials = args.trials or train_config['training']['optuna']['n_trials']
        print(f"Running optimization with {optuna_trials} trials...")
        
        # Quick mode: single trial with current best params
        if args.quick:
            print("⚡ Quick mode: Running single trial with default parameters")
            optuna_trials = 1
        
        study = optuna.create_study(direction=train_config['training']['optuna']['direction'])
        study.optimize(objective, n_trials=optuna_trials)

        print(f"\n✓ Optimization completed!")
        print(f"Best trial: {study.best_trial.number}")
        print(f"Best validation loss: {study.best_value:.6f}")
        print(f"Best parameters: {study.best_trial.params}")

        # Log optimization results
        mlflow_tracker.log_metrics({
            "best_trial_number": study.best_trial.number,
            "best_validation_loss": study.best_value,
            "optimization_trials_completed": len(study.trials)
        })
        
        mlflow_tracker.log_params({
            "best_trial_params": study.best_trial.params
        })

        # Load best model with multiple fallback options
        best_model = define_model(study.best_trial)
        
        # Try to load model state from multiple sources
        model_loaded = False
        
        # Option 1: From trial user attributes
        if "best_model_state" in study.best_trial.user_attrs:
            try:
                best_model.load_state_dict(study.best_trial.user_attrs["best_model_state"])
                print("✓ Best model state loaded from trial user attributes")
                model_loaded = True
            except Exception as e:
                print(f"⚠ Failed to load from trial attributes: {e}")
        
        # Option 2: From global best model state
        if not model_loaded and best_global_model_state is not None:
            try:
                best_model.load_state_dict(best_global_model_state)
                print("✓ Best model state loaded from global storage")
                model_loaded = True
            except Exception as e:
                print(f"⚠ Failed to load from global storage: {e}")
        
        # Option 3: Retrain the best configuration
        if not model_loaded:
            print("⚠ Warning: No saved model state found. Re-training best configuration...")
            loss_fn = LossFactory.create_loss(model_config)
            optimizer = torch.optim.Adam(best_model.parameters(), lr=study.best_trial.params.get('learning_rate', 0.001))
            
            print("Re-training best model for 20 epochs...")
            for epoch in range(20):
                train_loss = train(best_model, data, optimizer, loss_fn)
                val_loss, direction_accuracy, mae = validate(best_model, data, loss_fn)
                if epoch % 5 == 0:
                    print(f"  Epoch {epoch+1}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}, Dir Acc = {direction_accuracy:.3f}")
        
        print(f"\n=== Stage 2: Evaluation on EVAL Data (Train→Eval) ===")
        stage2_metrics, stage2_df, stage2_plots = evaluate_financial_performance(
            best_model, train_data_path, eval_data_path, "train_to_eval"
        )
        mlflow_tracker.log_metrics(stage2_metrics)
        
        if stage2_df is not None:
            # Save evaluation results - use CLI output dir if specified
            save_dir = args.output_dir or os.path.join(BASE_DIR, "financial_kg_ds/experiments/evaluations")
            os.makedirs(save_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            stage2_csv_path = f"{save_dir}/train_to_eval_{timestamp}.csv"
            stage2_df.to_csv(stage2_csv_path, index=False)
            mlflow_tracker.log_artifact(stage2_csv_path)
            
            if stage2_plots is not None:
                stage2_plot_path = f"{save_dir}/train_to_eval_plots_{timestamp}.html"
                stage2_plots.write_html(stage2_plot_path)
                mlflow_tracker.log_artifact(stage2_plot_path)
        
        print(f"✓ Stage 2 completed - Key metrics:")
        for key, value in stage2_metrics.items():
            if 'direction_accuracy' in key or 'total_return' in key or 'win_rate' in key:
                print(f"  {key}: {value}")

        print(f"\n=== Stage 3: Final Evaluation on TEST Data (Eval→Test) ===")  
        stage3_metrics, stage3_df, stage3_plots = evaluate_financial_performance(
            best_model, eval_data_path, test_data_path, "eval_to_test"
        )
        mlflow_tracker.log_metrics(stage3_metrics)
        
        if stage3_df is not None:
            stage3_csv_path = f"{save_dir}/eval_to_test_{timestamp}.csv"
            stage3_df.to_csv(stage3_csv_path, index=False)
            mlflow_tracker.log_artifact(stage3_csv_path)
            
            if stage3_plots is not None:
                stage3_plot_path = f"{save_dir}/eval_to_test_plots_{timestamp}.html"
                stage3_plots.write_html(stage3_plot_path)
                mlflow_tracker.log_artifact(stage3_plot_path)
        
        print(f"✓ Stage 3 completed - Key metrics:")
        for key, value in stage3_metrics.items():
            if 'direction_accuracy' in key or 'total_return' in key or 'win_rate' in key:
                print(f"  {key}: {value}")

        # Save final model - use CLI output dir if specified
        model_save_dir = args.output_dir or os.path.join(BASE_DIR, "financial_kg_ds/data")
        os.makedirs(model_save_dir, exist_ok=True)
        model_name = f"best_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        model_path = f"{model_save_dir}/{model_name}.pth"
        
        torch.save({
            'model_state_dict': best_model.state_dict(),
            'model_params': study.best_trial.params,
            'validation_loss': study.best_value,
            'stage2_metrics': stage2_metrics,
            'stage3_metrics': stage3_metrics
        }, model_path)
        
        # Log model to MLflow
        mlflow_tracker.log_model(model=best_model, name=model_name, data=data)
        mlflow_tracker.log_artifact(model_path)
        
        print(f"\n✓ Model saved to: {model_path}")
        print(f"✓ All results logged to MLflow experiment: {mlflow_tracker.experiment_name}")
        
        # Final summary
        print(f"\n=== TRAINING SUMMARY ===")
        print(f"Best model validation loss: {study.best_value:.6f}")
        if 'train_to_eval_direction_accuracy' in stage2_metrics:
            print(f"Train→Eval direction accuracy: {stage2_metrics['train_to_eval_direction_accuracy']:.3f}")
        if 'eval_to_test_direction_accuracy' in stage3_metrics:
            print(f"Eval→Test direction accuracy: {stage3_metrics['eval_to_test_direction_accuracy']:.3f}")
        
        return best_model, study, stage2_metrics, stage3_metrics

    except Exception as e:
        print(f"Training failed with error: {e}")
        mlflow_tracker.log_metrics({"training_failed": 1.0})
        raise
    finally:
        mlflow_tracker.end_run()

if __name__ == "__main__":
    main()
    
# Support module execution: python -m financial_kg_ds.train.train_node_regression
if __name__ == "__main__" or (hasattr(sys, '_getframe') and sys._getframe(1) is None):
    if len(sys.argv) > 1 and sys.argv[0].endswith('train_node_regression.py'):
        main()

