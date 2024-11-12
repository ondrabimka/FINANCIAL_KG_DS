# %%
from json import load
from financial_kg_ds.models.RNN_autoencoder import LSTMAutoencoder
from financial_kg_ds.datasets.download_data import HistoricalData
from financial_kg_ds.utils.paths import HISTORICAL_DATA_FILE
from financial_kg_ds.datasets.rnn_loader import RNNLoader
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import optuna


# %% make historical data exist otherwise download it
tickers = list(pd.read_csv(os.getenv('DATA_PATH') + '/ticker_info.csv', usecols=['ticker'])['ticker'])
# if not os.path.exists(f'{HISTORICAL_DATA_FILE}/historical_data.csv'):
# historical_data = HistoricalData(tickers)
# historical_data.download_data(period='2y', interval='1h')

# %% Prepare data
data_df = pd.read_csv(f'{HISTORICAL_DATA_FILE}/historical_data.csv', index_col=0)
data_df = data_df[data_df.index <= '2024-09-06'] # date based on FINANCIAL_KG data
# %%
data_df = data_df.fillna(0)

# %%
data_df = data_df[['Open_A','Volume_A','Open_AA','Open_AA']]


# %%
window_size = 49
batch_size = 32
shuffle = True
loader = RNNLoader.ae_from_dataframe(data_df[:500], window_size, batch_size, shuffle)

# %%
X, _ = next(iter(loader))

# %%
data_df[:500]

# %%
X.shape

# %%
def define_model(trial):
    lstm_layers = trial.suggest_int('lstm_layers', 1, 3)
    hidden_size = trial.suggest_int('hidden_size', 6, 32)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    return LSTMAutoencoder(window_size, lstm_layers, hidden_size, dropout)

def objective(trial):
    model = define_model(trial)
    criterion = nn.MSELoss()
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'RMSprop', 'SGD'])
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    for _ in range(150):
        for X, _ in loader:
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, X)
            loss.backward()
            optimizer.step()
    return loss.item()

# %%
def callback(study, trial):
    if study.best_trial.number == trial.number:
        study.set_user_attr(key="best_model", value=trial.user_attrs["best_model"])

# %%
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=2, callbacks=[callback])

# %%
best_model = study.best_trial.user_attrs['best_model']
torch.save(best_model.state_dict(), 'best_model.pth')


# %%
data_df[['Open_A']].head(30)
# %%
data_df
# %%