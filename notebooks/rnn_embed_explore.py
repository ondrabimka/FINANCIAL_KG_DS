
# %% AE embedding exploration
from financial_kg_ds.datasets.encoders import TimeSeriesEncoder
import pandas as pd
import numpy as np

from financial_kg_ds.utils.paths import HISTORICAL_DATA_FILE

# %% load data
data_df = pd.read_csv(f"{HISTORICAL_DATA_FILE}/historical_data_1h.csv", index_col=0, nrows=1000)
data_df = data_df[data_df.index <= "2024-11-12"]
data_df = data_df[[col for col in data_df.columns if "Close" in col]] # %% keep all Close_ticker columns
data_df = data_df.tail(50)
data_df = data_df.dropna(axis=1) # %% drop na columnwise

# %%
data_df

# %%
encoder = TimeSeriesEncoder("financial_kg_ds/data/best_model_bidi_29_1_2025-01-08.pth")

# %%
transoforemed_list = []

for col in data_df.columns:
    transformed = encoder(data_df[[col]])
    transoforemed_list.append({"ticker": col, "transformed": transformed.detach().numpy()})

# %% transoforemed_list to dataframe
transformed_df = pd.DataFrame(transoforemed_list)
transformed_df

# %% reduce dimensionality
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca.fit(np.concatenate(transformed_df["transformed"].values))
transformed_df["pca"] = transformed_df["transformed"].apply(lambda x: pca.transform(x).flatten())

# %%
transformed_df["pca_x"] = transformed_df["pca"].apply(lambda x: x[0])
transformed_df["pca_y"] = transformed_df["pca"].apply(lambda x: x[1])

# %% plot pca with plotly where x and y are pca[0] and pca[1] respectively
import plotly.express as px

fig = px.scatter(transformed_df, x="pca_x", y="pca_y", hover_data=["ticker"])
fig.show()

# %% plot GIL and CTGO close from data_df
# import matplotlib.pyplot as plt
# 
# plt.plot(data_df["Close_DTC"])
# plt.show()
# # %%
# 
# plt.plot(data_df["Close_STTK"])
# plt.show()
# # %%
