# %%
import matplotlib.pyplot as plt
from torch_geometric.transforms import ToUndirected

from financial_kg_ds.datasets.graph_loader import GraphLoader

data = GraphLoader.get_data()
data = ToUndirected()(data)
data

import torch

# %%
from torch_geometric.nn import MetaPath2Vec

metapath = [
    ("news", "about_nt", "ticker"),
    ("ticker", "holds_mt", "mutual_fund"),
    ("mutual_fund", "rev_holds_mt", "ticker"),
    ("ticker", "holds_it", "institution"),
    ("institution", "rev_holds_it", "ticker"),
    ("ticker", "rev_about_nt", "news"),
]

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# %%
data.edge_index_dict

# %%
model = MetaPath2Vec(
    data.edge_index_dict,
    embedding_dim=32,
    metapath=metapath,
    walk_length=8,
    context_size=7,
    walks_per_node=5,
    num_negative_samples=5,
    sparse=True,
).to(device)

# %%
z_ticker = model("ticker").detach().numpy()
z_institution = model("institution").detach().numpy()
z_mutual_fund = model("mutual_fund").detach().numpy()
z_news = model("news").detach().numpy()

# %%
z_ticker

# %% reduce the dimensionality of the embeddings with PCA
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
z_ticker_2d = pca.fit_transform(z_ticker)
z_institution_2d = pca.fit_transform(z_institution)
z_mutual_fund_2d = pca.fit_transform(z_mutual_fund)
z_news_2d = pca.fit_transform(z_news)

# %% plot the embeddings
plt.figure(figsize=(10, 10))
plt.scatter(z_ticker_2d[:, 0], z_ticker_2d[:, 1], s=10, alpha=0.5, label="Ticker")
plt.scatter(z_institution_2d[:, 0], z_institution_2d[:, 1], s=10, alpha=0.5, label="Institution")
plt.scatter(z_mutual_fund_2d[:, 0], z_mutual_fund_2d[:, 1], s=10, alpha=0.5, label="Mutual Fund")
plt.scatter(z_news_2d[:, 0], z_news_2d[:, 1], s=10, alpha=0.5, label="News")
plt.legend()
plt.show()

# %%
loader = model.loader(batch_size=128, shuffle=True)
optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)


def train(epoch, log_steps=100):
    model.train()

    total_loss = 0
    for i, (pos_rw, neg_rw) in enumerate(loader):
        optimizer.zero_grad()
        loss = model.loss(pos_rw.to(device), neg_rw.to(device))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if (i + 1) % log_steps == 0:
            print(f"Epoch: {epoch}, Step: {i + 1:05d}/{len(loader)}, " f"Loss: {total_loss / log_steps:.4f}")
            total_loss = 0


for epoch in range(1, 1000):
    train(epoch)

# %%
z_ticker = model("ticker").detach().numpy()
z_institution = model("institution").detach().numpy()
z_mutual_fund = model("mutual_fund").detach().numpy()
z_news = model("news").detach().numpy()

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
z_ticker_2d = pca.fit_transform(z_ticker)
z_institution_2d = pca.fit_transform(z_institution)
z_mutual_fund_2d = pca.fit_transform(z_mutual_fund)
z_news_2d = pca.fit_transform(z_news)

# %% plot the embeddings
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
plt.scatter(z_ticker_2d[:, 0], z_ticker_2d[:, 1], s=10, alpha=0.8, label="Ticker")
plt.scatter(z_institution_2d[:, 0], z_institution_2d[:, 1], s=10, alpha=0.8, label="Institution")
plt.scatter(z_mutual_fund_2d[:, 0], z_mutual_fund_2d[:, 1], s=10, alpha=0.8, label="Mutual Fund")
plt.scatter(z_news_2d[:, 0], z_news_2d[:, 1], s=10, alpha=0.8, label="News")
plt.legend()
plt.show()
