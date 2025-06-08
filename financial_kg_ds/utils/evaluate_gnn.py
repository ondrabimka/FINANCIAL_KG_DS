# %%
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import os
import json
import torch


class ModelEvaluator:
    def __init__(self, threshold=100, prediction_limit=5):
        self.threshold = threshold
        self.prediction_limit = prediction_limit
        
    def evaluate(self, model, data):
        """Comprehensive model evaluation including trading metrics and statistical tests"""
        model.eval()
        with torch.no_grad():
            # Get predictions
            out = model(data.x_dict, data.edge_index_dict)
            mask = data["ticker"].test_mask
            predictions = out[mask].cpu().numpy()
            true_values = data["ticker"].y[mask].cpu().numpy()
            tickers = data["ticker"].name[mask] if hasattr(data["ticker"], "name") else None
            
            # Create evaluation DataFrame
            df_eval = self._create_evaluation_df(predictions, true_values, tickers)
            df_clean = self._clean_data(df_eval)
            
            # Calculate all metrics
            metrics = self._calculate_metrics(df_clean)
            
            # Generate and save plots
            fig = self._create_evaluation_plots(df_clean)
            
            return metrics, df_clean, fig
        
    
    def _create_evaluation_df(self, predictions, true_values, tickers):
        df = pd.DataFrame({
            'ticker': tickers,
            'prediction': predictions.squeeze(),
            'actual': true_values.squeeze()
        })
        # Add derived columns
        df['return_actual'] = (df['actual'] - df['prediction']) / df['prediction'] * 100
        df['predicted_direction'] = df['prediction'].apply(lambda x: 1 if x > 0 else -1)
        df['actual_direction'] = df['actual'].apply(lambda x: 1 if x > 0 else -1)
        return df
        
    def _clean_data(self, df):
        df_clean = df.dropna()
        return df_clean[
            (df_clean['return_actual'].abs() < self.threshold) & 
            (df_clean['prediction'].abs() < self.prediction_limit)
        ]
    
    def _calculate_metrics(self, df):
        # Standard metrics
        mse = mean_squared_error(df['actual'], df['prediction'])
        metrics = {
            'test_mse': float(mse),  # Convert numpy types to Python types
            'test_rmse': float(np.sqrt(mse)),
            'test_mae': float(mean_absolute_error(df['actual'], df['prediction'])),
            'test_r2': float(r2_score(df['actual'], df['prediction'])),
            'market_return': float(df['return_actual'].mean()),
            'median_return': float(df['return_actual'].median()),
            'direction_accuracy': float((df['predicted_direction'] == df['actual_direction']).mean()),
            'profitable_trades': float((df['return_actual'] > 0).mean()),
            'total_predictions': int(len(df)),  # Convert to int
        }
        
        # Add statistical tests
        correlation, p_value = stats.pearsonr(df['prediction'], df['actual'])
        metrics.update({
            'correlation': float(correlation),
            'p_value': float(p_value),
            'statistically_significant': bool(p_value < 0.05)  # Convert to bool
        })
        
        return metrics
    
    def _create_evaluation_plots(self, df):
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Distribution of Returns', 'Predicted vs Actual Returns')
        )

        # Return distribution plot
        market_return = df['return_actual'].mean()
        fig.add_trace(
            go.Histogram(x=df['return_actual'], nbinsx=50, name='Returns'),
            row=1, col=1
        )
        fig.add_vline(x=market_return, line_dash="dash", line_color="red",
                    annotation_text=f"Market Avg: {market_return:.2f}%", row=1, col=1)
        fig.add_vline(x=0, line_dash="dash", line_color="green",
                    annotation_text="Zero Return", row=1, col=1)

        # Scatter plot
        fig.add_trace(
            go.Scatter(x=df['prediction'], y=df['return_actual'],
                    mode='markers', name='Predictions',
                    hovertemplate="Ticker: %{customdata}<br>Prediction: %{x:.2f}<br>Actual Return: %{y:.2f}%",
                    customdata=df['ticker']),
            row=1, col=2
        )

        # Update layout
        fig.update_layout(
            height=600,
            width=1200,
            showlegend=True,
            title_text="Model Evaluation Results"
        )

        return fig
    
    def save_results(self, df, metrics, fig, save_dir):
        """Save evaluation results to disk"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        os.makedirs(save_dir, exist_ok=True)
        
        # Save DataFrame
        df.to_csv(f"{save_dir}/evaluation_{timestamp}.csv", index=False)
        
        # Save metrics
        with open(f"{save_dir}/metrics_{timestamp}.json", 'w') as f:
            json.dump(metrics, f, indent=4)
        
        # Save plot
        fig.write_html(f"{save_dir}/plots_{timestamp}.html")
        
        return {
            'df_path': f"{save_dir}/evaluation_{timestamp}.csv",
            'metrics_path': f"{save_dir}/metrics_{timestamp}.json",
            'plot_path': f"{save_dir}/plots_{timestamp}.html"
        }
    

# %%
from financial_kg_ds.datasets.graph_loader import GraphLoaderRegresion
from torch_geometric.transforms import ToUndirected
from financial_kg_ds.models.GNN_hetero_sage_conv import HeteroGNN

data = GraphLoaderRegresion.get_data()
data = ToUndirected()(data)

# %%

# model = HeteroGNN(data.metadata(), hidden_channels=16, out_channels=1, num_layers=2)
# out = model(data.x_dict, data.edge_index_dict)

# %%
import mlflow
from mlflow.models import Model

model_uri = 'runs:/729d772ff60f4bfd9db483b02e75757b/best_model_20250605_172810'
# The model is logged with an input example
pyfunc_model = mlflow.pyfunc.load_model(model_uri)

# Verify the model with the provided input data using the logged dependencies.
# For more details, refer to:
# https://mlflow.org/docs/latest/models.html#validate-models-before-deployment
mlflow.models.predict(
    model_uri=model_uri,
    input_data=data,
    env_manager="uv",
)