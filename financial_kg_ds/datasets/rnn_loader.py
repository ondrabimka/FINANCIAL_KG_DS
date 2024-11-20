import torch
from torch.utils.data import Dataset, DataLoader
from typing import List
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class TimeseriesDataset(Dataset):

    """ Custom pytorch Dataset class

    More information: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
    """

    def __init__(self, X, y):

        """
        Parameters
        ----------
        X: tensor
            Training values
        y: tensor
            Label values
        """
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return (self.X[index], self.y[index])
    

class RNNLoader:

    def __init__(self, X, y, batch_size=32, shuffle=True):
        self.dataset = TimeseriesDataset(X, y)
        self.batch_size = batch_size
        self.shuffle = shuffle

    def get_loader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=self.shuffle)
        
    @classmethod
    def ae_from_dataframe(cls, df: pd.DataFrame, window_size: int = 30, batch_size: int = 32, shuffle: bool = True, device = torch.device("cpu"), scaler = MinMaxScaler()) -> DataLoader:
        """ Create a DataLoader from a dataframe for an autoencoder model

        Parameters
        ----------
        df: pd.DataFrame
            Dataframe containing the data

        window_size: int
            Size of the window for the model

        batch_size: int
            Size of the batch

        shuffle: bool
            Shuffle the data

        device: torch.device
            Device to use

        scaler: MinMaxScaler
            Scaler to use for the data. If None, no scaling is done. Default is MinMaxScaler(). Scaling is done in every window separately.

        Returns
        -------
        DataLoader
            DataLoader for the model

        Example
        -------
        >>> df = pd.read_csv('data.csv')
        df
        [[Close_AAPL, Volume_AAPL, Close_AMZN, Volume_AMZN],
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16],
        [17, 18, 19, 20]]
        >>> loader = RNNLoader.ae_from_dataframe(df, 3, 2, False)
        >>> X, _ = next(iter(loader))
        X
        [[[1, 2], 
        [5, 6], 
        [9, 10]], 
        [[3, 4], 
        [7, 8], 
        [11, 12]]]

        We don't need to specify the y values because they are the same as the X values
        """
        X = []
        y = []
        tickers = cls._extract_tickers_from_cols(df)
        for ticker in tickers:
            ticker_df = df.filter(regex=f'_({ticker})$')
            ticker_df = cls._keep_cols_names(ticker_df)
            for i in range(len(ticker_df) - window_size + 1):
                if scaler:
                    X.append(scaler.fit_transform(ticker_df.iloc[i:i + window_size].values))
                else:
                    X.append(ticker_df.iloc[i:i + window_size].values)
                y.append([0]) # y is the same as X for an autoencoder
        X = torch.from_numpy(np.array(X)).float().to(device)
        y = torch.from_numpy(np.array(y)).float().to(device)
        return cls(X, y, batch_size=batch_size, shuffle=shuffle).get_loader()

    @staticmethod
    def _extract_tickers_from_cols(df: pd.DataFrame) -> List[str]:
        """ Extract tickers from columns. Keep original order """
        return list(dict.fromkeys([col.split('_')[1] for col in df.columns]))
    
    @staticmethod
    def _keep_cols_names(df: pd.DataFrame, col_names: List[str] = ['Open','Volume']):
        """ Keep columns that contain any of the col_names in their name (Close, Volume, etc) """
        return df[[col for col in df.columns if any([col_name in col for col_name in col_names])]]