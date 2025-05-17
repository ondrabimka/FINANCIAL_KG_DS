# %%
import os
from datetime import date
from json import load

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from financial_kg_ds.datasets.historical_data import HistoricalData
from financial_kg_ds.datasets.rnn_loader import RNNLoader
from financial_kg_ds.models.BiRNN_autoencoder import LSTMAutoencoderBidi
from financial_kg_ds.utils.utils import ALL_TICKERS
from sklearn.preprocessing import StandardScaler

from datetime import datetime


# %% TRAIN PARAMETERS
PERIOD = "10y"
INTERVAL = "1wk"
DATE_CUT_OFF = "2024-09-06"  # max date to consider for training
N_EPOCHS = 100

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

# %% Load historical data
historical_data = HistoricalData(ALL_TICKERS, period=PERIOD, interval=INTERVAL)
data_df = historical_data.combine_ticker_data(['Close'])

# %%
DATE_CUT_OFF = pd.to_datetime(DATE_CUT_OFF)
data_df = data_df[data_df.index.tz_localize(None) <= DATE_CUT_OFF]

# %%
window_size = 52
batch_size = 32
shuffle = True
rnn_loader = RNNLoader(
    data_df,
    window_size=window_size,
    batch_size=batch_size,
    shuffle=shuffle,
    scaler=StandardScaler(),
    cols_to_keep=["Close"],
    device=device,
)


# %%
train_loader, val_loader, test_loader = rnn_loader.get_loaders()

# %%
print(len(train_loader))
print(len(val_loader))
print(len(test_loader))

# %%
del data_df


# %%
def define_model(trial):
    lstm_layers = trial.suggest_int("lstm_layers", 1, 3)
    hidden_size = trial.suggest_int("hidden_size", 6, 128)
    dropout = trial.suggest_float("dropout", 0.01, 0.5)
    return LSTMAutoencoderBidi(1, hidden_size, lstm_layers, dropout).to(device)


def objective(trial):
    model = define_model(trial)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=trial.suggest_float("lr", 1e-5, 1e-2, log=True),
        weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    )
    
    val_loss_min = float('inf')
    patience = trial.suggest_int("patience", 5, 15)
    
    counter = 0
    for _ in range(N_EPOCHS):
        # Training
        model.train()
        train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            X = batch[0]
            out = model(X)
            loss = criterion(out, X)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                X = batch[0]
                out = model(X)
                val_loss += criterion(out, X).item()

        val_loss = val_loss / len(val_loader)
        
        if val_loss < val_loss_min:
            val_loss_min = val_loss
            trial.set_user_attr(key="best_model", value=model)
            print('Validation loss decreased ({:.6f} --> {:.6f})'.format(val_loss_min, val_loss))
            counter = 0
        else:
            counter += 1
            
        if counter >= patience:
            break
            
    return val_loss_min


# %%
def callback(study, trial):
    if study.best_trial.number == trial.number:
        study.set_user_attr(key="best_model", value=trial.user_attrs["best_model"])


# %%
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=3, callbacks=[callback])

# %%
print("Best trial")
print("  Value: ", study.best_trial.value)
print("  Params: ")
for key, value in study.best_trial.params.items():
    print("    {}: {}".format(key, value))

# %%
best_model = study.user_attrs["best_model"]

hidden_size = study.best_trial.params["hidden_size"]
num_layers = study.best_trial.params["lstm_layers"]
today = date.today().strftime("%Y-%m-%d")
date_cutoff = str(pd.to_datetime(DATE_CUT_OFF, format="%Y-%m-%d").date())

torch.save(best_model.state_dict(), f"financial_kg_ds/data/best_model_bidi_{hidden_size}_{num_layers}_{today}_{PERIOD}_{INTERVAL}_{date_cutoff}.pth")


# %% plot data
import matplotlib.pyplot as plt


def plot_data(data, model):
    with torch.no_grad():
        # model.eval()
        output = model(torch.unsqueeze(data, 0))
        plt.plot(data.cpu().numpy().flatten(), label="data")
        plt.plot(output.cpu().numpy().flatten(), label="output")
        plt.legend()
        plt.show()


# %%
X = next(iter(test_loader))

output = best_model(torch.unsqueeze(X[0][0], 0))
output
# %%
output[0][:, [0]]

# %%
timeseries_pred = output[0][:, [0]]
# timeseries_volume = output[0][:, [1]]
# %%
plt.plot(X[0][0][:, [0]].cpu().numpy().flatten(), label="data")
plt.plot(timeseries_pred.cpu().detach().numpy().flatten(), label="pred")
plt.legend()
plt.show()

# %%
# plt.plot(X[0][0][:,[1]].cpu().numpy().flatten(), label='data')
# plt.plot(timeseries_volume.cpu().detach().numpy().flatten(), label="pred")
# plt.legend()
# plt.show()

