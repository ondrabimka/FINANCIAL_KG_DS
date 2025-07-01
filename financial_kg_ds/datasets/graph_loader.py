# %%
import os
from typing import List

import pandas as pd
import torch
from dotenv import load_dotenv
from typing import Optional
from torch_geometric.data import Data, HeteroData

from financial_kg_ds.datasets.encoders import IdentityEncoder, OneHotEncoder, SentimentAnalysisEncoder, TimeSeriesEncoder

from financial_kg_ds.utils.utils import ALL_TICKERS

load_dotenv()

# TODO: #4 Refactor so that GraphLoaderBase loads the basic graph structure
class GraphLoaderBase:
    def __init__(self, data_path=os.getenv("TRAIN_DATA_PATH")):
        """
        Parameters
        ----------
        data_path : str
            Path to the data directory.
        """
        self.data_path = data_path
        self.data = HeteroData()

    def load_node_csv(self, path, col_to_map: Optional[str] = None, node_name_col: Optional[str] = None, encoders = None, **kwargs):
        """
        Load node data from csv file.

        Parameters
        ----------
        path : str
            Path to the csv file.

        col_to_map : str
            Column name to map. If None, the index is used. (default: None)

        node_name_col : str
            Column name of the node name. If None, the node name is not loaded. (default: None)

        encoders : dict
            Dictionary of encoders. (default: None)

        Returns
        -------
        x : torch.Tensor
            Node attribute tensor.

        mapping : dict
            Mapping from the index to the node index.
        """
        df = pd.read_csv(path, **kwargs)

        # if col_to_map is ticker make sure its filled with ALL_TICKERS
        if col_to_map == "ticker":
            ticker_df = pd.DataFrame({'ticker': ALL_TICKERS})
            df = ticker_df.merge(df, on='ticker', how='left')


        if col_to_map is not None:
            mapping = {index: i for i, index in enumerate(df[col_to_map].unique())}
        else:
            mapping = {index: i for i, index in enumerate(df.index)}

        x = None
        if encoders is not None:
            xs = [encoder(df[col]) for col, encoder in encoders.items()]
            x = torch.cat(xs, dim=-1)

        node_names = None
        if node_name_col is not None:
            node_names = df[node_name_col].unique()
            assert len(node_names) == x.shape[0], f"Number of node names ({len(node_names)}) does not match the number of nodes ({x.shape[0]})"

        return x, mapping, node_names

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

        if src_index_col is not None: src_to_map = df[src_index_col] 
        else: src_to_map = df.index

        if dst_index_col is not None: dst_to_map = df[dst_index_col]
        else:dst_to_map = df.index

        # skip if the source or destination index is not in the mapping
        src = [src_mapping.get(index, -1) for index in src_to_map]
        dst = [dst_mapping.get(index, -1) for index in dst_to_map]

        edge_index = torch.tensor([src, dst])

        edge_attr = None
        if encoders is not None:
            edge_attrs = [encoder(df[col]) for col, encoder in encoders.items()]
            edge_attr = torch.cat(edge_attrs, dim=-1)

        # if edge_index contains -1, remove the corresponding column from edge_attr and edge_index
        if -1 in edge_index:
            mask = (edge_index != -1).all(dim=0)
            edge_index = edge_index[:, mask]
            edge_attr = edge_attr[mask]

        return edge_index, edge_attr
    
    # default node loading methods
    def add_ticker_node(self):
        """ Add ticker node to the graph. """
        ticker_x, ticker_mapping, node_names = self.load_node_csv(
            self.data_path + "/ticker_info.csv", "ticker", "ticker", encoders={"ticker": OneHotEncoder(), "currentPrice": IdentityEncoder()}
        )
        self.data["ticker"].x = ticker_x.float()
        self.ticker_mapping = ticker_mapping

        if node_names is not None:
            self.data["ticker"].name = node_names

    def add_mutual_fund_node(self):
        mutual_fund_x, mutual_fund_mapping, node_names = self.load_node_csv(
            self.data_path + "/mutual_fund.csv", "name", node_name_col="name", encoders={"name": OneHotEncoder()}, na_values={"value": 0}
        )
        self.data["mutual_fund"].x = mutual_fund_x.float()
        self.mutual_fund_mapping = mutual_fund_mapping

        if node_names is not None:
            self.data["mutual_fund"].name = node_names

    def add_institution_node(self):
        institution_x, institution_mapping, node_names = self.load_node_csv(
            self.data_path + "/institution.csv", "name", node_name_col="name", encoders={"name": OneHotEncoder()}, na_values={"value": 0}
        )
        self.data["institution"].x = institution_x.float()
        self.institution_mapping = institution_mapping

        if node_names is not None:
            self.data["institution"].name = node_names

    def add_news_node(self):
        news_x, news_mapping, node_names = self.load_node_csv(self.data_path + "/news.csv", "title", node_name_col="title", encoders={"title": OneHotEncoder()})
        self.data["news"].x = news_x.float()
        self.news_mapping = news_mapping

        if node_names is not None:
            self.data["news"].name = node_names

    def add_insider_holder_node(self):
        insider_holder_x, insider_holder_mapping, node_names = self.load_node_csv(
            self.data_path + "/insider_holder.csv", "name", node_name_col="name", encoders={"name": OneHotEncoder()}
        )
        self.data["insider_holder"].x = insider_holder_x.float()
        self.insider_holder_mapping = insider_holder_mapping

        if node_names is not None:
            self.data["insider_holder"].name = node_names

    def add_insider_transaction_node(self):
        insider_transaction_x, insider_transaction_mapping, node_names = self.load_node_csv(
            self.data_path + "/insider_transaction.csv", encoders={"shares": IdentityEncoder()}
        )
        self.data["insider_transaction"].x = insider_transaction_x.float()
        self.insider_transaction_mapping = insider_transaction_mapping

        if node_names is not None:
            self.data["insider_transaction"].name = node_names
        else:
            # use index as node name
            self.data["insider_transaction"].name = [str(i) for i in range(insider_transaction_x.shape[0])]

    # default edge loading methods
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

    def holds_iht(self):
        edge_index, edge_attr = self.load_edge_csv(
            self.data_path + "/insider_holder.csv",
            "name",
            self.insider_holder_mapping,
            "ticker",
            self.ticker_mapping,
            # encoders={"shares": IdentityEncoder()},
        )
        self.data["insider_holder", "holds_iht", "ticker"].edge_index = edge_index
        self.data["insider_holder", "holds_iht", "ticker"].edge_attr = edge_attr

    def created(self):
        # inider holder created insider transaction
        edge_index, edge_attr = self.load_edge_csv(
            self.data_path + "/insider_transaction.csv",
            "name",
            self.insider_holder_mapping,
            None,
            self.insider_transaction_mapping,
            encoders={"shares": IdentityEncoder()},
        )
        self.data["insider_holder", "created", "insider_transaction"].edge_index = edge_index
        self.data["insider_holder", "created", "insider_transaction"].edge_attr = edge_attr

    def involves(self):
        # insider transaction involves ticker
        edge_index, edge_attr = self.load_edge_csv(
            self.data_path + "/insider_transaction.csv",
            None,
            self.insider_transaction_mapping,
            "ticker",
            self.ticker_mapping,
        )
        self.data["insider_transaction", "involves", "ticker"].edge_index = edge_index
        self.data["insider_transaction", "involves", "ticker"].edge_attr = edge_attr

    def add_mask(self):
        # 40% of the data is used for training, 30% for validation, and 30% for testing
        n = self.data["ticker"].num_nodes
        train_mask = torch.zeros(n, dtype=torch.bool)
        val_mask = torch.zeros(n, dtype=torch.bool)
        test_mask = torch.zeros(n, dtype=torch.bool)
        n_train = int(0.4 * n)
        n_val = int(0.3 * n)
        train_mask[:n_train] = True
        val_mask[n_train : n_train + n_val] = True
        test_mask[n_train + n_val :] = True
        self.data["ticker"].train_mask = train_mask
        self.data["ticker"].val_mask = val_mask
        self.data["ticker"].test_mask = test_mask

    def load_full_graph(self, add_mask=True):
        self.add_ticker_node()
        self.add_mutual_fund_node()
        self.add_institution_node()
        self.add_news_node()
        self.add_insider_holder_node()
        self.add_insider_transaction_node()
        self.add_holds_it_rel()
        self.add_holds_mt_rel()
        self.about_nt_rel()
        self.holds_iht()
        self.created()
        self.involves()

        if add_mask:
            self.add_mask()
        
        return self.data
    
    @classmethod
    def get_data(cls, data_path=os.getenv("TRAIN_DATA_PATH")):
        """
        Get the graph data.
        """
        loader = cls(data_path)
        return loader.load_full_graph()

class GraphLoaderRegresion(GraphLoaderBase):
    def __init__(self, data_path=os.getenv("TRAIN_DATA_PATH"), eval_data_path=os.getenv("EVAL_DATA_PATH")):
        """
        Parameters
        ----------
        data_path : str
            Path to the data directory.

        eval_data_path : str
            Path to the data from which the labels are computed.
        """
        super().__init__(data_path)
        self.eval_data_path = eval_data_path

    # Upadate hetero data with node and edge data
    def add_ticker_node(self):
        ticker_x, ticker_mapping, node_names = self.load_node_csv(
            self.data_path + "/ticker_info.csv",
            col_to_map="ticker",
            node_name_col="symbol",
            encoders={
                # "ticker": OneHotEncoder(),
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
                "ticker": TimeSeriesEncoder("10y", "1wk", "2024-09-06"),
            },
        )
        # 'trailingPE': IdentityEncoder(),})
        self.data["ticker"].x = ticker_x.float()
        self.ticker_mapping = ticker_mapping

        if node_names is not None:
            self.data["ticker"].name = node_names

    def add_news_node(self):
        print("loading news")
        news_x, news_mapping, node_names = self.load_node_csv(
            self.data_path + "/news.csv", "title", node_name_col="title", encoders={"title": SentimentAnalysisEncoder()}
        )
        self.data["news"].x = news_x.float()
        self.news_mapping = news_mapping
        if node_names is not None:
            self.data["news"].name = node_names

    def load_label(self):

        """
        Load the label for the ticker node, which is the percentage change (scaled) in market cap between the training and test data.
        The label is computed as follows:
        1. Load the market cap from the ticker_info.csv file in both the training and test data.
        2. Compute the difference in market cap between the test and training data.
        3. Scale the difference by dividing it by the market cap in the training data and multiplying by 100.
        4. Clip the value to a maximum of 500 (500% from the original market cap).
        5. Scale the value using StandardScaler with mean set to False.
        6. Convert the value to a torch tensor and assign it to the ticker node's y attribute.
        """

        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler(with_mean=False)

        tickers = pd.read_csv(self.data_path + "/ticker_info.csv")
        tickers_new = pd.read_csv(self.eval_data_path + "/ticker_info.csv")
        tickers = tickers[["symbol", "marketCap"]]
        tickers_new = tickers_new[["symbol", "marketCap"]]
        tickers = tickers.drop_duplicates(subset="symbol")
        tickers_new = tickers_new.drop_duplicates(subset="symbol")
        tickers = tickers.set_index("symbol")
        tickers_new = tickers_new.set_index("symbol")
        ticker = tickers.merge(tickers_new, on="symbol", suffixes=("_old", "_new"), how="left")
        ticker["mcap_diff"] = ticker["marketCap_new"] - ticker["marketCap_old"]
        ticker["mcap_diff"] = ticker["mcap_diff"] / ticker["marketCap_old"] * 100
        ticker["mcap_diff"] = ticker["mcap_diff"].clip(upper=500)
        ticker["mcap_diff"] = ticker["mcap_diff"].fillna(0)
        ticker["mcap_diff"] = scaler.fit_transform(ticker["mcap_diff"].values.reshape(-1, 1))
        mcap_diff = torch.from_numpy(ticker["mcap_diff"].values).view(-1, 1)
        self.data["ticker"].y = mcap_diff.float()

    def load_full_graph(self, add_mask=True):
        self.add_ticker_node()
        self.add_mutual_fund_node()
        self.add_institution_node()
        self.add_news_node()
        self.add_insider_holder_node()
        self.add_insider_transaction_node()
        self.add_holds_it_rel()
        self.add_holds_mt_rel()
        self.about_nt_rel()
        self.holds_iht()
        self.created()
        self.involves()
        self.load_label()

        if add_mask:
            self.add_mask()
        
        return self.data
    
    @classmethod
    def get_data(cls, data_path=os.getenv("TRAIN_DATA_PATH"), eval_data_path=os.getenv("EVAL_DATA_PATH")):
        """
        Get the graph data.
        """
        loader = cls(data_path, eval_data_path)
        return loader.load_full_graph()