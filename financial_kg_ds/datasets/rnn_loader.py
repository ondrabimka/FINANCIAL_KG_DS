# %% 
from typing import List

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset
from typing import Tuple

class TimeseriesDataset(Dataset):

    """Custom pytorch Dataset class

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
    def __init__(
        self, 
        df: pd.DataFrame, 
        window_size: int = 30, 
        batch_size: int = 32,
        shuffle: bool = True, 
        scaler=None,
        cols_to_keep: List[str] = ["Close", "Volume"],
        device: torch.device = torch.device("cpu"),
        ):

        """
        Parameters
        ----------
        df: pd.DataFrame
            Dataframe containing the data

        window_size: int
            Size of the window to use for the data

        batch_size: int
            Size of the batch to use for the data

        shuffle: bool
            Whether to shuffle the data or not

        scaler: sklearn.preprocessing
            Scaler to use for the data

        cols_to_keep: List[str]
            List of columns to keep in the dataframe

        device: torch.device
            Device to use for the data
        """
        if len(df) < window_size:
            raise ValueError("Dataframe is too small")

        self.df = df
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.scaler = scaler
        self.window_size = window_size
        self.cols_to_keep = cols_to_keep
        self.device = device

    def dataframe_to_windows(self, df: pd.DataFrame = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert dataframe to windows of data. Data is scaled on every window separately.

        Parameters
        ----------
        df: pd.DataFrame
            Dataframe containing the data

        DataLoader
            DataLoader for the model

        Returns
        -------
        X: torch.Tensor
            Tensor containing the data

        y: torch.Tensor
            Tensor containing the labels

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
        >>> loader = RNNLoader.dataframe_to_windows(df)
        >>> X, _ = next(iter(loader))
        X
        [[[1, 2],
        [5, 6],
        [9, 10]],
        [[5, 6],
        [9, 10],
        [13, 14]],
        [[9, 10],
        [13, 14],
        [17, 18]]]
        [[[3, 4],
        [7, 8],
        [11, 12]],
        [[7, 8],
        [11, 12],
        [15, 16]],
        [[11, 12],
        [15, 16],
        [19, 20]]]
        ...
        """
        X = []
        y = []

        if df is None:
            df = self.df

        tickers = self._extract_tickers_from_cols(df)
        for ticker in tickers:
            ticker_df = df.filter(regex=f"_({ticker})$")
            ticker_df = self._keep_cols_names(ticker_df, cols_to_keep=self.cols_to_keep)
            for i in range(len(ticker_df) - self.window_size + 1):
                if self.scaler:
                    X.append(self.scaler.fit_transform(ticker_df.iloc[i : i + self.window_size].values))
                else:
                    X.append(ticker_df.iloc[i : i + self.window_size].values)
                y.append([0])  # y is the same as X for an autoencoder
        X = torch.from_numpy(np.array(X)).float().to(self.device)
        y = torch.from_numpy(np.array(y)).float().to(self.device)
        return X, y
    

    def get_loaders(
        self,
        val_split: float = 0.2,
        test_split: float = 0.2,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Get the data loaders for the training, validation and test sets.

        Parameters
        ----------
        val_split: float
            Fraction of the data to use for validation

        test_split: float
            Fraction of the data to use for testing

        Returns
        -------
        train_loader: DataLoader
            DataLoader for the training set

        val_loader: DataLoader
            DataLoader for the validation set

        test_loader: DataLoader
            DataLoader for the test set

        Notes
        -----
        We don't need to specify the y values because they are the same as the X values
        """

        # Split the data into training, validation and test sets
        train_size = int(len(self.df) * (1 - val_split - test_split))
        val_size = int(len(self.df) * val_split)

        train_df = self.df[:train_size]
        val_df = self.df[train_size : train_size + val_size]
        test_df = self.df[train_size + val_size :]

        train_windows = self.dataframe_to_windows(train_df)
        val_windows = self.dataframe_to_windows(val_df)
        test_windows = self.dataframe_to_windows(test_df)

        # Create the data loaders
        train_windows = TimeseriesDataset(*train_windows)
        val_windows = TimeseriesDataset(*val_windows)
        test_windows = TimeseriesDataset(*test_windows)

        train_loader = DataLoader(train_windows, batch_size=self.batch_size, shuffle=self.shuffle)
        val_loader = DataLoader(val_windows, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(test_windows, batch_size=self.batch_size, shuffle=False)

        return train_loader, val_loader, test_loader
    
    @staticmethod
    def _extract_tickers_from_cols(df: pd.DataFrame) -> List[str]:
        """Extract tickers from columns. Keep original order"""
        return list(dict.fromkeys([col.split("_")[1] for col in df.columns]))

    @staticmethod
    def _keep_cols_names(df: pd.DataFrame, cols_to_keep: List[str] = ["Close", "Volume"]):
        """Keep columns that contain any of the cols_to_keep in their name (Close, Volume, etc)"""
        return df[[col for col in df.columns if any([col_name in col for col_name in cols_to_keep])]]
