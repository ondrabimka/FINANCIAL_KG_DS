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

from financial_kg_ds.datasets.download_data import HistoricalData
from financial_kg_ds.datasets.rnn_loader import RNNLoader
from financial_kg_ds.models.RNN_autoencoder import LSTMAutoencoder
from financial_kg_ds.utils.paths import HISTORICAL_DATA_FILE

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

# %% make historical data exist otherwise download it
tickers = list(pd.read_csv(os.getenv("DATA_PATH") + "/ticker_info.csv", usecols=["ticker"])["ticker"])
# if not os.path.exists(f'{HISTORICAL_DATA_FILE}/historical_data.csv'):
# historical_data = HistoricalData(tickers)
# historical_data.download_data(period='2y', interval='1h')

# %% Prepare data
data_df = pd.read_csv(f"{HISTORICAL_DATA_FILE}/historical_data.csv", index_col=0)
data_df = data_df[data_df.index <= "2024-09-06"]  # date based on FINANCIAL_KG data
data_df = data_df.fillna(0)

# %%
window_size = 49
batch_size = 32
shuffle = True
loader = RNNLoader.ae_from_dataframe(data_df, window_size, batch_size, shuffle, device)

# %%
# import matplotlib.pyplot as plt
#
# def plot_tensor(tensor):
#     fig, axs = plt.subplots(2)
#     axs[0].scatter(range(tensor.shape[0]), tensor[:, 0])
#     axs[1].hist(tensor[:, 1])
#     plt.show()
#
# X = next(iter(loader))
# plot_tensor(X[0][0])

# %%
del data_df

# %%
def define_model(trial):
    lstm_layers = trial.suggest_int("lstm_layers", 1, 3)
    hidden_size = trial.suggest_int("hidden_size", 6, 64)
    dropout = trial.suggest_float("dropout", 0.01, 0.5)
    return LSTMAutoencoder(2, lstm_layers, hidden_size, dropout)


def objective(trial):
    model = define_model(trial).to(device)
    criterion = nn.MSELoss()
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    train_loss = float("inf")

    for epoch in range(20):
        for X, _ in loader:
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, X)
            loss.backward()
            optimizer.step()

        if loss < train_loss:
            train_loss = loss
            print("train loss decreased:", train_loss)
            trial.set_user_attr("best_model", value=model)

        trial.report(loss, epoch)
    return loss.item()


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
best_model = study.best_trial.user_attrs["best_model"]

hidden_size = study.best_trial.params["hidden_size"]
num_layers = study.best_trial.params["lstm_layers"]
today = date.today().strftime("%Y-%m-%d")

torch.save(best_model.state_dict(), f"financial_kg_ds/data/best_model_{hidden_size}_{num_layers}_{today}.pth")
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
X = next(iter(loader))

# %%
output = best_model(torch.unsqueeze(X[0][0], 0))
# %%
timeseries_pred = output[0][:, [0]]
timeseries_volume = output[0][:, [1]]
# %%
plt.plot(X[0][0][:, [0]].cpu().numpy().flatten(), label="data")
plt.plot(timeseries_pred.cpu().detach().numpy().flatten(), label="pred")
plt.legend()
plt.show()

# %%
# plt.plot(X[0][0][:,[1]].cpu().numpy().flatten(), label='data')
plt.plot(timeseries_volume.cpu().detach().numpy().flatten(), label="pred")
plt.legend()
plt.show()
# %%
