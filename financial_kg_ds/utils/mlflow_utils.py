import mlflow
import mlflow.pytorch
from mlflow.tracking import MlflowClient
import os
import json
import pandas as pd
import numpy as np
import torch
from datetime import datetime
import logging
import tempfile
from typing import Dict, Any, List, Optional

# Setup logger for this module
logger = logging.getLogger(__name__)

class MLflowTracker:
    def __init__(self, experiment_name: str, tracking_uri: str = "http://localhost:5001", registry_uri: str = "sqlite:///mlruns.db", tags: Dict[str, str] = None):
        self.experiment_name = experiment_name
        
        # Set tracking URI
        mlflow.set_tracking_uri(tracking_uri)
        logger.info(f"MLflow tracking URI: {tracking_uri}")
        
        # Set registry URI
        mlflow.set_registry_uri(registry_uri)
        logger.info(f"MLflow registry URI: {registry_uri}")
        
        # Default experiment tags
        default_tags = {
            "project": "financial_kg_ds",
            "framework": "pytorch"
        }
        if tags:
            default_tags.update(tags)
        
        # Create or get experiment
        try:
            self.experiment_id = mlflow.create_experiment(
                experiment_name,
                tags=default_tags
            )
            logger.info(f"Created new MLflow experiment: {experiment_name}")
        except mlflow.exceptions.MlflowException:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            self.experiment_id = experiment.experiment_id
            logger.info(f"Using existing MLflow experiment: {experiment_name}")
        
        mlflow.set_experiment(experiment_name)
        self.client = MlflowClient()

    def start_run(self, run_name: str = None, tags: Dict[str, str] = None, nested: bool = False) -> mlflow.ActiveRun:
        """
        Start a new MLflow run
        
        Parameters
        ----------
        run_name : str, optional
            Name for the run
        tags : Dict[str, str], optional
            Additional tags for the run
        nested : bool, optional
            Whether this is a nested run (default: False)
            
        Returns
        -------
        mlflow.ActiveRun
            Active MLflow run
        """
        if tags is None:
            tags = {}
        
        # Add default tags
        default_tags = {
            "created_at": datetime.now().isoformat()
        }
        tags.update(default_tags)
        
        active_run = mlflow.start_run(run_name=run_name, tags=tags, nested=nested)
        self.run_id = active_run.info.run_id
        return active_run

    def log_params(self, params: Dict[str, Any]):
        """
        Log hyperparameters to MLFlow
        
        Parameters
        ----------
        params : Dict[str, Any]
            Hyperparameters to log
        """
        # Convert complex objects to strings
        clean_params = {}
        for key, value in params.items():
            if isinstance(value, (dict, list)):
                clean_params[key] = json.dumps(value)
            elif isinstance(value, (np.integer, np.floating)):
                clean_params[key] = value.item()
            else:
                clean_params[key] = value
        
        mlflow.log_params(clean_params)

    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """
        Log metrics to MLFlow
        
        Parameters
        ----------
        metrics : Dict[str, float]
            Metrics to log
        step : int, optional
            Step number for the metrics
        """
        # Clean metrics (handle NaN values)
        clean_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, (np.integer, np.floating)):
                value = value.item()
            
            if np.isfinite(value):
                clean_metrics[key] = float(value)
            else:
                clean_metrics[key] = 0.0  # Replace NaN/inf with 0
        
        mlflow.log_metrics(clean_metrics, step=step)

    def log_model(self, model, name: str, data=None):
        """Log model with signature and input example"""
        if data is not None:
            # Create sample input with serializable keys
            sample_input = {
                'x_dict': {str(k): v[:1].detach().cpu().numpy() 
                          for k, v in data.x_dict.items()},
                'edge_index_dict': {str(k): v[:, :1].detach().cpu().numpy() 
                                   for k, v in data.edge_index_dict.items()}
            }
            
            # Define model signature
            from mlflow.models.signature import infer_signature
            prediction = model(data.x_dict, data.edge_index_dict)[:1].detach().cpu().numpy()
            signature = infer_signature(sample_input, prediction)
            
            # Log model with pytorch flavor
            registered_model_name = f"{self.experiment_name}_{name}"
            mlflow.pytorch.log_model(
                model,
                artifact_path=name,
                signature=signature,
                input_example=sample_input,
                registered_model_name=registered_model_name  # This registers the model
            )
        else:
            registered_model_name = f"{self.experiment_name}_{name}"
            mlflow.pytorch.log_model(
                model,
                artifact_path=name,
                registered_model_name=registered_model_name
            )

    def log_artifacts(self, local_dir: str, artifact_path: str = None):
        """
        Log artifacts directory to MLFlow
        
        Parameters
        ----------
        local_dir : str
            Local directory to log
        artifact_path : str, optional
            Artifact path in MLFlow
        """
        if os.path.exists(local_dir):
            mlflow.log_artifacts(local_dir, artifact_path)

    def log_figure(self, figure, filename: str):
        """
        Log matplotlib figure to MLFlow
        
        Parameters
        ----------
        figure : matplotlib.figure.Figure
            Figure to log
        filename : str
            Filename for the figure
        """
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            figure.savefig(tmp_file.name, dpi=150, bbox_inches='tight')
            mlflow.log_artifact(tmp_file.name, filename)

    def log_dataframe(self, df: pd.DataFrame, filename: str):
        """
        Log pandas DataFrame to MLFlow
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to log
        filename : str
            Filename for the CSV file
        """
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp_file:
            df.to_csv(tmp_file.name, index=False)
            mlflow.log_artifact(tmp_file.name, filename)

    def log_artifact(self, local_path: str):
        mlflow.log_artifact(local_path)

    def end_run(self):
        mlflow.end_run()