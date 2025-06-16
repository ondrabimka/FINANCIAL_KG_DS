import mlflow
import mlflow.pytorch
from typing import Dict, Any
import os

class MLflowTracker:
    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        # mlruns_dir = "mlruns"
        # os.makedirs(mlruns_dir, exist_ok=True)
        # self.mlflow_uri = "file://" + os.path.abspath(mlruns_dir)
        self.mlflow_uri = "http://localhost:5001"
        mlflow.set_tracking_uri(self.mlflow_uri)
        # mlflow.set_tracking_uri(self.mlflow_uri)
        
        # Create or get experiment
        try:
            self.experiment_id = mlflow.create_experiment(experiment_name)
        except mlflow.exceptions.MlflowException:
            self.experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

        self.model_registry_uri = "sqlite:///mlruns.db"
        mlflow.set_registry_uri(self.model_registry_uri)

    def start_run(self, run_name: str, nested: bool = False):
        """Start a new MLflow run
        
        Args:
            run_name: Name of the run
            nested: Whether this is a nested run
        """
        active_run = mlflow.start_run(
            experiment_id=self.experiment_id, 
            run_name=run_name,
            nested=nested
        )
        self.run_id = active_run.info.run_id
        return self.run_id

    def log_params(self, params: Dict[str, Any]):
        for key, value in params.items():
            mlflow.log_param(key, value)

    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)

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

    def log_artifact(self, local_path: str):
        mlflow.log_artifact(local_path)

    def end_run(self):
        mlflow.end_run()