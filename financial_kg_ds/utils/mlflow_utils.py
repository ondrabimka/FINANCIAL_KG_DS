import mlflow
import mlflow.pytorch
from typing import Dict, Any
import os

class MLflowTracker:
    def __init__(self, experiment_name: str):
        self.experiment_name = experiment_name
        mlruns_dir = "mlruns"
        os.makedirs(mlruns_dir, exist_ok=True)
        # self.mlflow_uri = "file://" + os.path.abspath(mlruns_dir)
        mlflow.set_tracking_uri("http://localhost:5000")
        # mlflow.set_tracking_uri(self.mlflow_uri)
        
        # Create or get experiment
        try:
            self.experiment_id = mlflow.create_experiment(
                experiment_name,
                artifact_location=os.path.join(self.mlflow_uri, experiment_name)
            )
        except:
            self.experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

    def start_run(self, run_name: str):
        active_run = mlflow.start_run(experiment_id=self.experiment_id, run_name=run_name)
        self.run_id = active_run.info.run_id
        return self.run_id

    def log_params(self, params: Dict[str, Any]):
        for key, value in params.items():
            mlflow.log_param(key, value)

    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)

    def log_model(self, model, name: str):
        mlflow.pytorch.log_model(model, name)

    def log_artifact(self, local_path: str):
        mlflow.log_artifact(local_path)

    def end_run(self):
        mlflow.end_run()