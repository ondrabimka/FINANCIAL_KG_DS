import yfinance as yf
from typing import List
import os
import pandas as pd
from time import sleep

# TODO: Extend to download missing parts if date or ticker is missing
class HistoricalData:

    """
    Class to download historical data from Yahoo Finance API
    
    Parameters
    ----------
    tickers : List[str]
        List of tickers to download data from
        
    Methods
    -------
    download_data(**kwargs)
        Download historical data from Yahoo Finance API
        
    load_data(**kwargs)
        Load historical data from Yahoo Finance API
    """

    def __init__(self, tickers: List[str]):
        self.tickers = [yf.Ticker(ticker) for ticker in tickers]
            
    def download_data(self, **kwargs):
        """
        Download historical data from Yahoo Finance API
        
        Parameters
        ----------
        **kwargs : dict
            Keyword arguments to pass to yfinance.Ticker.history method and sleep time between requests
        """
        dfs = []
        for ticker in self.tickers:
            df = ticker.history(**kwargs)
            df.columns = [f"{col}_{ticker.ticker}" for col in df.columns]
            print(f"Downloaded data for {ticker.ticker}")
            if df.empty or len(df) == 0:
                print(f"Ticker {ticker.ticker} is not valid")
            dfs.append(df)
            sleep(kwargs.get("sleep", 1))
        self._validate_data_dir()
        pd.concat(dfs, axis=1).to_csv("data/historical_data.csv")

    def load_data(self, **kwargs):
        """
        Load historical data from Yahoo Finance API
        
        Parameters
        ----------
        **kwargs : dict
            Keyword arguments to pass to yfinance.Ticker.history method and sleep time between requests
        """
        self.download_data(**kwargs)
        return self._load_file()[0]

    def _load_file(self):
        historical_data = pd.read_csv("data/historical_data.csv")
        tickers_already_downloaded = historical_data.columns.str.split("_").str[-1].unique()
        dates_already_downloaded = historical_data.index
        return historical_data, tickers_already_downloaded, dates_already_downloaded
    
    def _validate_data_dir(self):
        if not os.path.exists("data"):
            os.makedirs("data")
