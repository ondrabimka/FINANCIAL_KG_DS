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
from financial_kg_ds.utils.utils import ALL_TICKERS

# %%
class ModelEvaluator:
    def __init__(self, start_data_path, end_data_path, threshold=100, prediction_limit=5):
        self.start_data_path = f"{start_data_path}/ticker_info.csv"
        self.end_data_path = f"{end_data_path}/ticker_info.csv"
        self.threshold = threshold
        self.prediction_limit = prediction_limit
        self.price_data = self.load_price_data(self.start_data_path, self.end_data_path)

    def evaluate(self, model, data):
        model.eval()
        with torch.no_grad():
            out = model(data.x_dict, data.edge_index_dict)
            predictions = out.cpu().numpy().squeeze()
            tickers = data["ticker"].name

            # make sure tickers and ALL_TICKERS are aligned
            if tickers is None or len(tickers) == 0:
                raise ValueError("No tickers found in the test mask. Ensure the model and data are correctly set up.")
            if len(tickers) != len(ALL_TICKERS):
                raise ValueError(f"Mismatch in number of tickers: expected {len(ALL_TICKERS)}, got {len(tickers)}")

            # Prepare DataFrame with tickers and predictions
            df_pred = pd.DataFrame({
                'ticker': tickers,
                'predicted_scaled_mcap_change': predictions
            })

            # Merge with price_data to get unscaled values
            df = pd.merge(df_pred, self.price_data, on='ticker', how='left')

            # Calculate predicted and actual market cap and price
            df['actual_mcap_change'] = (df['end_marketCap'] - df['start_marketCap']) / df['start_marketCap']
            df['predicted_end_marketCap'] = df['start_marketCap'] * (1 + df['predicted_scaled_mcap_change'])
            df['predicted_mcap_change'] = (df['predicted_end_marketCap'] - df['start_marketCap']) / df['start_marketCap']

            # If you want to estimate predicted price change proportional to mcap change:
            df['predicted_end_price'] = df['start_currentPrice'] * (df['predicted_end_marketCap'] / df['start_marketCap'])
            df['actual_return'] = (df['end_currentPrice'] - df['start_currentPrice']) / df['start_currentPrice'] * 100
            df['predicted_return'] = (df['predicted_end_price'] - df['start_currentPrice']) / df['start_currentPrice'] * 100

            df['predicted_direction'] = df['predicted_return'].apply(lambda x: 1 if x > 0 else -1)
            df['actual_direction'] = df['actual_return'].apply(lambda x: 1 if x > 0 else -1)

            # Clean data
            df_clean = df.dropna()
            df_clean = df_clean[
                (df_clean['actual_return'].abs() < self.threshold) &
                (df_clean['predicted_return'].abs() < self.prediction_limit)
            ]

            # Calculate metrics
            metrics = self._calculate_metrics(df_clean)

            # Generate and save plots
            fig = self._create_evaluation_plots(df_clean)

            return metrics, df_clean, fig

    def _calculate_metrics(self, df):
        # Regression metrics on returns
        mse = mean_squared_error(df['actual_return'], df['predicted_return'])
        metrics = {
            'test_mse': float(mse),
            'test_rmse': float(np.sqrt(mse)),
            'test_mae': float(mean_absolute_error(df['actual_return'], df['predicted_return'])),
            'test_r2': float(r2_score(df['actual_return'], df['predicted_return'])),
            'market_return': float(df['actual_return'].mean()),
            'median_return': float(df['actual_return'].median()),
            'direction_accuracy': float((df['predicted_direction'] == df['actual_direction']).mean()),
            'profitable_trades': float((df['predicted_return'] > 0).mean()),
            'total_predictions': int(len(df)),
        }
        # Statistical tests
        if len(df) > 1:
            correlation, p_value = stats.pearsonr(df['predicted_return'], df['actual_return'])
        else:
            correlation, p_value = np.nan, np.nan
        metrics.update({
            'correlation': float(correlation) if not np.isnan(correlation) else None,
            'p_value': float(p_value) if not np.isnan(p_value) else None,
            'statistically_significant': bool(p_value < 0.05) if not np.isnan(p_value) else False
        })
        return metrics

    def _create_evaluation_plots(self, df):
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Distribution of Actual Returns', 'Predicted vs Actual Returns')
        )
        market_return = df['actual_return'].mean()
        fig.add_trace(
            go.Histogram(x=df['actual_return'], nbinsx=50, name='Actual Returns'),
            row=1, col=1
        )
        fig.add_vline(x=market_return, line_dash="dash", line_color="red",
                      annotation_text=f"Market Avg: {market_return:.2f}%", row=1, col=1)
        fig.add_vline(x=0, line_dash="dash", line_color="green",
                      annotation_text="Zero Return", row=1, col=1)
        fig.add_trace(
            go.Scatter(x=df['predicted_return'], y=df['actual_return'],
                       mode='markers', name='Predictions',
                       hovertemplate="Ticker: %{customdata}<br>Predicted: %{x:.2f}<br>Actual: %{y:.2f}%",
                       customdata=df['ticker']),
            row=1, col=2
        )
        fig.update_layout(
            height=600,
            width=1200,
            showlegend=True,
            title_text="Model Evaluation Results"
        )
        return fig

    def save_results(self, df, metrics, fig, save_dir):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        os.makedirs(save_dir, exist_ok=True)
        df.to_csv(f"{save_dir}/evaluation_{timestamp}.csv", index=False)
        with open(f"{save_dir}/metrics_{timestamp}.json", 'w') as f:
            json.dump(metrics, f, indent=4)
        fig.write_html(f"{save_dir}/plots_{timestamp}.html")
        return {
            'df_path': f"{save_dir}/evaluation_{timestamp}.csv",
            'metrics_path': f"{save_dir}/metrics_{timestamp}.json",
            'plot_path': f"{save_dir}/plots_{timestamp}.html"
        }

    @staticmethod
    def extract_date_from_path(path):
        name = path.split('/')[-1]
        date_str = name.split('_')[1]
        return datetime.strptime(date_str, '%Y-%m-%d').date()

    def load_price_data(self, start_data_path, end_data_path):
        data_start = pd.read_csv(start_data_path)
        data_end = pd.read_csv(end_data_path)
        data_start = data_start[['ticker', 'marketCap', 'currentPrice', 'lastSplitDate']]
        data_start = data_start.rename(columns={'marketCap': 'start_marketCap', 'currentPrice': 'start_currentPrice', 'lastSplitDate': 'start_lastSplitDate'})
        data_end = data_end[['ticker', 'marketCap', 'currentPrice', 'lastSplitDate']]
        data_end = data_end.rename(columns={'marketCap': 'end_marketCap', 'currentPrice': 'end_currentPrice', 'lastSplitDate': 'end_lastSplitDate'})
        ticker_df = pd.DataFrame(ALL_TICKERS, columns=['ticker'])
        data_merged = ticker_df.merge(data_start, on='ticker', how='left')\
                               .merge(data_end, on='ticker', how='left')
        data_merged['start_lastSplitDate'] = data_merged['start_lastSplitDate'].fillna(0)
        data_merged['end_lastSplitDate'] = data_merged['end_lastSplitDate'].fillna(0)
        # Remove rows with splits between periods
        data_merged.loc[data_merged['start_lastSplitDate'] != data_merged['end_lastSplitDate'],
                        ['start_marketCap', 'start_currentPrice', 'end_marketCap', 'end_currentPrice']] = None
        data_merged = data_merged.drop(columns=['start_lastSplitDate', 'end_lastSplitDate'])
        return data_merged
                

'''
# %%
from financial_kg_ds.datasets.graph_loader import GraphLoaderRegresion
from torch_geometric.transforms import ToUndirected
from financial_kg_ds.models.GNN_hetero_sage_conv import HeteroGNN
import torch

data = GraphLoaderRegresion.get_data()
data = ToUndirected()(data)

# %%

model_path = 'mlruns/1/729d772ff60f4bfd9db483b02e75757b/artifacts/best_model_20250605_172810/data/model.pth'
model = torch.load(model_path, map_location=torch.device('cpu'))
out = model(data.x_dict, data.edge_index_dict)
# %%
import os

start_data_path = os.getenv("EVAL_DATA_PATH")
end_data_path = os.getenv("EVAL_DATA_PATH")

evaluator = ModelEvaluator(
    start_data_path=start_data_path,
    end_data_path=end_data_path,
    threshold=100,
    prediction_limit=5
)

# %%
from financial_kg_ds.utils.utils import ALL_TICKERS
import pandas as pd

df = pd.DataFrame({'ticker': ALL_TICKERS})
ticker_info = df.merge(ticker_info, on='ticker', how='left')
start_data_path = os.getenv("EVAL_DATA_PATH")

ticker_info = pd.read_csv(f"{start_data_path}/ticker_info.csv")
# %%
df

# %%
ticker_info = df.merge(ticker_info, on='ticker', how='left')
# %%
ticker_info
# %%
'''