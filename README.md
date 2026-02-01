# FINANCIAL_KG_DS

Welcome to the FINANCIAL_KG_DS repository. This project is a data science extension to https://github.com/ondrabimka/FINANCIAL_KG.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Training](#training)
- [Evaluation](#evaluation)
- [MLflow Integration](#mlflow-integration)

## Introduction
The FINANCIAL_KG_DS project aims to provide tools and resources for building and analyzing financial knowledge graphs. It includes data processing scripts, graph algorithms, and visualization tools.

## Installation
To get started, clone the repository and install the required dependencies:

```bash
git clone https://github.com/ondrabimka/FINANCIAL_KG_DS.git
cd FINANCIAL_KG_DS
pip install -r requirements.txt

# Create necessary directories
mkdir -p configs/{models,training}
mkdir -p mlruns
mkdir -p financial_kg_ds/experiments/evaluations
```

Create .env file in the root directory and add the following environment variables:
```bash
DATA_PATH='../FINANCIAL_KG/data/data_2024-09-06'
```

## Project Structure
```
FINANCIAL_KG_DS/
├── configs/
│   ├── models/           # Model configurations
│   └── training/         # Training configurations
├── financial_kg_ds/
│   ├── datasets/         # Data loading and processing
│   ├── models/          # Model architectures
│   ├── train/          # Training scripts
│   ├── utils/          # Utility functions
│   └── experiments/    # Experiment results
├── mlruns/             # MLflow tracking
└── README.md
```

## Configuration
The project uses YAML configuration files for models and training:

### Model Configuration
```yaml
# configs/models/base_gnn.yaml
model:
  name: "HeteroGNN"
  fixed_params:
    out_channels: 1
    dropout: 0.2
  optuna_params:
    hidden_channels:
      min: 32
      max: 256
    # ...more parameters...

# News encoder configuration (NEW!)
news_encoder:
  type: "sentiment"  # Options: sentiment (FinBERT), openai (OpenAI embeddings), onehot
  openai_model: "text-embedding-3-small"  # Used when type="openai"

loss:
  name: "asymmetric"  # Options: mse, asymmetric, huber, quantile
  params:
    alpha: 1.5
```

**News Encoder Options:**
- `sentiment`: FinBERT sentiment analysis (default) - 2D features
- `openai`: OpenAI embeddings - 1536 or 3072D rich semantic features
- `onehot`: One-hot encoding - baseline

For detailed news encoder configuration, see [docs/NEWS_ENCODER_GUIDE.md](docs/NEWS_ENCODER_GUIDE.md)

### Training Configuration
```yaml
# configs/training/default_training.yaml
training:
  num_epochs: 100
  early_stopping:
    patience: 5
    min_delta: 1e-4
  # ...more parameters...
```

## Training
To train a model:
```bash
# With default sentiment encoder
python -m financial_kg_ds.train.train_node_regression

# With OpenAI embeddings (requires OPENAI_API_KEY env var)
python -m financial_kg_ds.train.train_node_regression --news-encoder openai

# With custom configuration
python -m financial_kg_ds.train.train_node_regression \
  --news-encoder openai \
  --openai-model text-embedding-3-large \
  --epochs 200 \
  --trials 30
```

**Using OpenAI Embeddings:**
```bash
# Set your OpenAI API key
export OPENAI_API_KEY='your-api-key-here'  # Linux/Mac
set OPENAI_API_KEY=your-api-key-here       # Windows

# Or add to .env file
echo "OPENAI_API_KEY=your-api-key-here" >> .env
```

Note: MLFlow needs to be running to track experiments. You can start it with: [MLflow Integration](#mlflow-integration)

## Evaluation
The system provides comprehensive evaluation metrics including:
- Standard metrics (MSE, RMSE, MAE, R²)
- Trading metrics (Direction accuracy, Profitable trades)
- Market-adjusted metrics (Information ratio)
- Statistical analysis

## MLflow Integration
View training results and experiments:
```bash
mlflow server --backend-store-uri sqlite:///mlruns.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5001
```
Then open http://localhost:5000 in your browser.

For more detailed documentation, see the respective README files in each directory.
