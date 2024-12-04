# %%
import os
from typing import List

import pandas as pd
import torch
from dotenv import load_dotenv
from torch_geometric.data import Data, HeteroData

from financial_kg_ds.datasets.encoders import IdentityEncoder, OneHotEncoder, SentimentAnalysisEncoder

load_dotenv()


class GraphLoaderBase:
    def __init__(self, data_path=os.getenv("DATA_PATH")):
        """
        Parameters
        ----------
        data_path : str
            Path to the data directory.
        """
        self.data_path = data_path
        self.data = HeteroData()

    def load_node_csv(self, path, col_to_map: str, encoders=None, **kwargs):
        """
        Load node data from csv file.

        Parameters
        ----------
        path : str
            Path to the csv file.

        col_to_map : str
            Column name to map.

        encoders : dict
            Dictionary of encoders.

        Returns
        -------
        x : torch.Tensor
            Node attribute tensor.

        mapping : dict
            Mapping from the index to the node index.
        """
        df = pd.read_csv(path, **kwargs)
        mapping = {index: i for i, index in enumerate(df[col_to_map].unique())}

        x = None
        if encoders is not None:
            xs = [encoder(df[col]) for col, encoder in encoders.items()]
            x = torch.cat(xs, dim=-1)

        return x, mapping

    def load_edge_csv(self, path, src_index_col, src_mapping, dst_index_col, dst_mapping, encoders=None, **kwargs):
        """
        Load edge data from csv file.

        Parameters
        ----------
        path : str
            Path to the csv file.

        src_index_col : str
            Column name of the source index.

        src_mapping : dict
            Mapping from the source index to the node index.

        dst_index_col : str
            Column name of the destination index.

        dst_mapping : dict
            Mapping from the destination index to the node index.

        encoders : dict
            Dictionary of encoders.

        Returns
        -------
        edge_index : torch.Tensor
            Edge index tensor.

        edge_attr : torch.Tensor
            Edge attribute tensor.
        """
        df = pd.read_csv(path, **kwargs)

        src = [src_mapping[index] for index in df[src_index_col]]
        dst = [dst_mapping[index] for index in df[dst_index_col]]
        edge_index = torch.tensor([src, dst])

        edge_attr = None
        if encoders is not None:
            edge_attrs = [encoder(df[col]) for col, encoder in encoders.items()]
            edge_attr = torch.cat(edge_attrs, dim=-1)

        return edge_index, edge_attr


class GraphLoaderRegresion(GraphLoaderBase):
    def __init__(self, data_path=os.getenv("DATA_PATH")):
        """
        Parameters
        ----------
        data_path : str
            Path to the data directory.
        """
        super().__init__(data_path)

    # Upadate hetero data with node and edge data
    def add_ticker_node(self):
        ticker_x, ticker_mapping = self.load_node_csv(
            self.data_path + "/ticker_info.csv",
            "ticker",
            encoders={
                "ticker": OneHotEncoder(),
                "currentPrice": IdentityEncoder(),
                "targetHighPrice": IdentityEncoder(),
                "targetLowPrice": IdentityEncoder(),
                "targetMeanPrice": IdentityEncoder(),
                "targetMedianPrice": IdentityEncoder(),
                "recommendationMean": IdentityEncoder(),
                "marketCap": IdentityEncoder(),
                # 'sector': OneHotEncoder(),
                "totalCash": IdentityEncoder(),
                "totalDebt": IdentityEncoder(),
                "shortRatio": IdentityEncoder(),
                "overallRisk": IdentityEncoder(),
                "payoutRatio": IdentityEncoder(),
                "priceHint": IdentityEncoder(),
                "returnOnAssets": IdentityEncoder(),
                "returnOnEquity": IdentityEncoder(),
                "freeCashflow": IdentityEncoder(),
                "insidersPercentHeld": IdentityEncoder(),
                "institutionsPercentHeld": IdentityEncoder(),
            },
        )
        # 'trailingPE': IdentityEncoder(),})
        self.data["ticker"].x = ticker_x.float()
        self.ticker_mapping = ticker_mapping

    def add_mutual_fund_node(self):
        mutual_fund_x, mutual_fund_mapping = self.load_node_csv(
            self.data_path + "/mutual_fund.csv", "name", encoders={"name": OneHotEncoder()}, na_values={"value": 0}
        )
        self.data["mutual_fund"].x = mutual_fund_x.float()
        self.mutual_fund_mapping = mutual_fund_mapping

    def add_institution_node(self):
        institution_x, institution_mapping = self.load_node_csv(
            self.data_path + "/institution.csv", "name", encoders={"name": OneHotEncoder()}, na_values={"value": 0}
        )
        self.data["institution"].x = institution_x.float()
        self.institution_mapping = institution_mapping

    def add_news_node(self):
        print("loading news")
        news_x, news_mapping = self.load_node_csv(
            self.data_path + "/news.csv", "title", encoders={"title": SentimentAnalysisEncoder()}
        )
        self.data["news"].x = news_x.float()
        self.news_mapping = news_mapping

    def add_holds_it_rel(self):
        edge_index, edge_attr = self.load_edge_csv(
            self.data_path + "/institution.csv",
            "ticker",
            self.ticker_mapping,
            "name",
            self.institution_mapping,
            encoders={"value": IdentityEncoder(), "pctHeld": IdentityEncoder()},
        )
        self.data["ticker", "holds_it", "institution"].edge_index = edge_index
        self.data["ticker", "holds_it", "institution"].edge_attr = edge_attr

    def add_holds_mt_rel(self):
        edge_index, edge_attr = self.load_edge_csv(
            self.data_path + "/mutual_fund.csv",
            "ticker",
            self.ticker_mapping,
            "name",
            self.mutual_fund_mapping,
            encoders={"value": IdentityEncoder(), "pctHeld": IdentityEncoder()},
        )
        self.data["ticker", "holds_mt", "mutual_fund"].edge_index = edge_index
        self.data["ticker", "holds_mt", "mutual_fund"].edge_attr = edge_attr

    def about_nt_rel(self):
        edge_index, edge_attr = self.load_edge_csv(
            self.data_path + "/news.csv", "title", self.news_mapping, "ticker", self.ticker_mapping
        )
        self.data["news", "about_nt", "ticker"].edge_index = edge_index
        self.data["news", "about_nt", "ticker"].edge_attr = edge_attr

    def add_mask(self):
        # 20% of the data is used for training, 20% for validation, and 60% for testing
        n = self.data["ticker"].num_nodes
        train_mask = torch.zeros(n, dtype=torch.bool)
        val_mask = torch.zeros(n, dtype=torch.bool)
        test_mask = torch.zeros(n, dtype=torch.bool)
        n_train = int(0.3 * n)
        n_val = int(0.3 * n)
        train_mask[:n_train] = True
        val_mask[n_train : n_train + n_val] = True
        test_mask[n_train + n_val :] = True
        self.data["ticker"].train_mask = train_mask
        self.data["ticker"].val_mask = val_mask
        self.data["ticker"].test_mask = test_mask

    def load_label(self):
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler(with_mean=False)

        tickers = pd.read_csv("/Users/obimka/Desktop/Zabafa/FINANCIAL_KG/data/data_2024-09-06/ticker_info.csv")
        tickers_new = pd.read_csv("/Users/obimka/Desktop/Zabafa/FINANCIAL_KG/data/data_2024-10-03/ticker_info.csv")
        tickers = tickers[["symbol", "marketCap"]]
        tickers_new = tickers_new[["symbol", "marketCap"]]
        tickers = tickers.drop_duplicates(subset="symbol")
        tickers_new = tickers_new.drop_duplicates(subset="symbol")
        tickers = tickers.set_index("symbol")
        tickers_new = tickers_new.set_index("symbol")
        ticker = tickers.merge(tickers_new, on="symbol", suffixes=("_old", "_new"), how="left")
        ticker["mcap_diff"] = ticker["marketCap_new"] - ticker["marketCap_old"]
        ticker["mcap_diff"] = ticker["mcap_diff"] / ticker["marketCap_old"] * 100
        ticker["mcap_diff"] = ticker["mcap_diff"].clip(upper=200)
        ticker["mcap_diff"] = ticker["mcap_diff"].fillna(0)
        ticker["mcap_diff"] = scaler.fit_transform(ticker["mcap_diff"].values.reshape(-1, 1))
        mcap_diff = torch.from_numpy(ticker["mcap_diff"].values).view(-1, 1)
        self.data["ticker"].y = mcap_diff.float()

    def load_full_graph(self):
        self.add_ticker_node()
        self.add_mutual_fund_node()
        self.add_institution_node()
        self.add_news_node()
        self.add_holds_it_rel()
        self.add_holds_mt_rel()
        self.about_nt_rel()
        self.add_mask()
        self.load_label()
        return self.data

    @classmethod
    def get_data(cls, data_path=os.getenv("DATA_PATH")):
        """
        Get the graph data.
        """
        loader = cls(data_path)
        return loader.load_full_graph()


class GraphLoaderAE(GraphLoaderBase):
    def __init__(self, data_path=os.getenv("DATA_PATH")):
        """
        Parameters
        ----------
        data_path : str
            Path to the data directory.
        """
        super().__init__(data_path)

    # Upadate hetero data with node and edge data
    def add_ticker_node(self):
        ticker_x, ticker_mapping = self.load_node_csv(
            self.data_path + "/ticker_info.csv", "ticker", encoders={"ticker": OneHotEncoder(), "currentPrice": IdentityEncoder()}
        )  # ,'targetMeanPrice': IdentityEncoder(), 'marketCap': IdentityEncoder()})
        self.data["ticker"].x = ticker_x.float()
        self.ticker_mapping = ticker_mapping

    def add_mutual_fund_node(self):
        mutual_fund_x, mutual_fund_mapping = self.load_node_csv(
            self.data_path + "/mutual_fund.csv", "name", encoders={"name": OneHotEncoder()}
        )
        self.data["mutual_fund"].x = mutual_fund_x.float()
        self.mutual_fund_mapping = mutual_fund_mapping

    def add_institution_node(self):
        institution_x, institution_mapping = self.load_node_csv(
            self.data_path + "/institution.csv", "name", encoders={"name": OneHotEncoder()}
        )
        self.data["institution"].x = institution_x.float()
        self.institution_mapping = institution_mapping

    def add_news_node(self):
        news_x, news_mapping = self.load_node_csv(self.data_path + "/news.csv", "uuid", encoders={"uuid": OneHotEncoder()})
        self.data["news"].x = news_x.float()
        self.news_mapping = news_mapping

    def add_holds_it_rel(self):
        edge_index, edge_attr = self.load_edge_csv(
            self.data_path + "/institution.csv",
            "ticker",
            self.ticker_mapping,
            "name",
            self.institution_mapping,
            encoders={"value": IdentityEncoder()},
        )
        self.data["ticker", "holds_it", "institution"].edge_index = edge_index
        self.data["ticker", "holds_it", "institution"].edge_attr = edge_attr

    def add_holds_mt_rel(self):
        edge_index, edge_attr = self.load_edge_csv(
            self.data_path + "/mutual_fund.csv",
            "ticker",
            self.ticker_mapping,
            "name",
            self.mutual_fund_mapping,
            encoders={"value": IdentityEncoder()},
        )
        self.data["ticker", "holds_mt", "mutual_fund"].edge_index = edge_index
        self.data["ticker", "holds_mt", "mutual_fund"].edge_attr = edge_attr

    def about_nt_rel(self):
        edge_index, edge_attr = self.load_edge_csv(
            self.data_path + "/news.csv", "uuid", self.news_mapping, "ticker", self.ticker_mapping
        )
        self.data["news", "about_nt", "ticker"].edge_index = edge_index
        self.data["news", "about_nt", "ticker"].edge_attr = edge_attr

    def load_full_graph(self):
        self.add_ticker_node()
        self.add_mutual_fund_node()
        self.add_institution_node()
        self.add_news_node()
        self.add_holds_it_rel()
        self.add_holds_mt_rel()
        self.about_nt_rel()
        return self.data

    @classmethod
    def get_data(cls, data_path=os.getenv("DATA_PATH")):
        """
        Get the graph data.
        """
        loader = cls(data_path)
        return loader.load_full_graph()
