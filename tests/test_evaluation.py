import unittest
import numpy as np
import pandas as pd
import torch
import tempfile
import os
from financial_kg_ds.utils.evaluate_gnn import ModelEvaluator
from torch_geometric.data import HeteroData

class MockModel:
    def __init__(self, predictions):
        self.predictions = predictions
        self.eval_called = False
    
    def eval(self):
        self.eval_called = True
    
    def __call__(self, x_dict, edge_index_dict):
        return self.predictions

class TestModelEvaluator(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method"""
        self.evaluator = ModelEvaluator(threshold=100, prediction_limit=5)
        
        # Create mock data
        self.data = HeteroData()
        self.data["ticker"].test_mask = torch.tensor([True] * 5)
        self.data["ticker"].y = torch.tensor([1.0, -1.0, 0.5, -0.5, 2.0])
        self.data["ticker"].name = ["AAPL", "GOOGL", "MSFT", "AMZN", "META"]
        
        # Create mock model
        predictions = torch.tensor([[0.8], [-0.9], [0.4], [-0.6], [1.8]])
        self.model = MockModel(predictions)

    def test_evaluator_initialization(self):
        """Test evaluator initialization with custom parameters"""
        evaluator = ModelEvaluator(threshold=50, prediction_limit=3)
        self.assertEqual(evaluator.threshold, 50)
        self.assertEqual(evaluator.prediction_limit, 3)

    def test_create_evaluation_df(self):
        """Test creation of evaluation DataFrame"""
        with torch.no_grad():
            predictions = self.model(None, None).numpy()
            true_values = self.data["ticker"].y.numpy()
            tickers = self.data["ticker"].name
        
        df = self.evaluator._create_evaluation_df(predictions, true_values, tickers)
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(
            set(df.columns), 
            {'ticker', 'prediction', 'actual', 'return_actual', 
             'predicted_direction', 'actual_direction'}
        )
        self.assertEqual(len(df), 5)

    def test_clean_data(self):
        """Test data cleaning with thresholds"""
        df = pd.DataFrame({
            'return_actual': [-150, -50, 0, 50, 150],
            'prediction': [-6, -2, 0, 2, 6]
        })
        
        df_clean = self.evaluator._clean_data(df)
        self.assertEqual(len(df_clean), 3)  # Only values within thresholds

    def test_calculate_metrics(self):
        """Test metric calculation"""
        df = pd.DataFrame({
            'actual': [1.0, -1.0, 0.5, -0.5],
            'prediction': [0.8, -0.9, 0.4, -0.6],
            'return_actual': [10, -10, 5, -5],
            'predicted_direction': [1, -1, 1, -1],
            'actual_direction': [1, -1, 1, -1]
        })
        
        metrics = self.evaluator._calculate_metrics(df)
        
        # Check metric types
        self.assertIsInstance(metrics['test_mse'], float)
        self.assertIsInstance(metrics['market_return'], float)
        self.assertIsInstance(metrics['total_predictions'], int)
        self.assertIsInstance(metrics['statistically_significant'], bool)
        
        # Check metric values
        self.assertEqual(metrics['direction_accuracy'], 1.0)  # Perfect direction prediction
        self.assertEqual(metrics['total_predictions'], 4)
        
    def test_full_evaluation(self):
        """Test complete evaluation pipeline"""
        # Update mock data with required attributes
        self.data["ticker"].x = torch.randn(5, 3)  # 5 nodes, 3 features
        self.data["ticker"].edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])  # Sample edges
        
        # Convert to dictionary format expected by HeteroGNN
        self.data.x_dict = {"ticker": self.data["ticker"].x}
        self.data.edge_index_dict = {
            ("ticker", "to", "ticker"): self.data["ticker"].edge_index
        }
        
        metrics, df_clean, fig = self.evaluator.evaluate(self.model, self.data)
        
        # Check if model.eval() was called
        self.assertTrue(self.model.eval_called)
        
        # Check outputs
        self.assertIsInstance(metrics, dict)
        self.assertIsInstance(df_clean, pd.DataFrame)
        self.assertIsNotNone(fig)  # Plotly figure
        
        # Check specific metrics are present
        expected_metrics = {'test_mse', 'market_return', 'direction_accuracy', 
                        'profitable_trades', 'total_predictions'}
        self.assertTrue(expected_metrics.issubset(metrics.keys()))

    def test_save_results(self):
        """Test saving evaluation results to files"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            df = pd.DataFrame({'test': [1, 2, 3]})
            metrics = {'metric1': 1.0, 'metric2': 2.0}
            fig = None  # Mock figure
            
            result_paths = self.evaluator.save_results(df, metrics, fig, tmp_dir)
            
            # Check file extensions
            self.assertTrue(result_paths['df_path'].endswith('.csv'))
            self.assertTrue(result_paths['metrics_path'].endswith('.json'))
            self.assertTrue(result_paths['plot_path'].endswith('.html'))
            
            # Check if files exist
            self.assertTrue(os.path.exists(result_paths['df_path']))
            self.assertTrue(os.path.exists(result_paths['metrics_path']))

if __name__ == '__main__':
    unittest.main()