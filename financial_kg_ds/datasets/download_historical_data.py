# %%
import os
from time import sleep
from typing import List

import pandas as pd
import yfinance as yf

from financial_kg_ds.utils.paths import HISTORICAL_DATA_FILE
from financial_kg_ds.utils.utils import ALL_TICKERS

class HistoricalData:
    def __init__(self, tickers: List[str], period: str = "2y", interval: str = "1wk"):
        self.tickers = [yf.Ticker(ticker) for ticker in tickers]
        self.period = period
        self.interval = interval
        self.data_dir = os.path.join(HISTORICAL_DATA_FILE, interval)
        self._validate_data_dir()

    def download_data(self, **kwargs):
        """Download historical data and save each ticker separately"""
        for ticker in self.tickers:
            ticker_path = os.path.join(self.data_dir, f"{ticker.ticker}.csv")
            
            # Skip if file exists
            if os.path.exists(ticker_path):
                print(f"Data for {ticker.ticker} already exists")
                continue
                
            df = ticker.history(period=self.period, interval=self.interval, **kwargs)
            if df.empty or len(df) == 0:
                print(f"Ticker {ticker.ticker} is not valid")
                continue
                
            print(f"Downloaded data for {ticker.ticker}")
            df.to_csv(ticker_path)
            sleep(kwargs.get("sleep", 1))

    def load_data(self, **kwargs):
        """Load all ticker data and combine into one DataFrame"""
        if not os.path.exists(self.data_dir):
            print("Data not downloaded yet")
            self.download_data(period=self.period, interval=self.interval, **kwargs)
        
        return self.combine_ticker_data()


    def combine_ticker_data(self, columns: List[str] = None):
        """ Combine all ticker CSV files into one DataFrame """

        assert isinstance(columns, list) or columns is None, "columns should be a list or None"
        valid_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
        if columns:
            for col in columns:
                assert col in valid_columns, f"Invalid column: {col}. Valid columns are: {valid_columns}"
            columns = ['Date'] + columns
        else:
            columns = valid_columns

        whole_df = pd.DataFrame()

        for ticker in self.tickers:
            ticker_path = os.path.join(self.data_dir, f"{ticker.ticker}.csv")
            if os.path.exists(ticker_path):
                df = pd.read_csv(ticker_path, usecols=columns)
                df.rename(columns={col: f"{col}_{ticker.ticker}" for col in df.columns if col != 'Date'}, inplace=True)
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                df = df.set_index('Date')
                print(df.head())

                if whole_df.empty:
                    whole_df = df
                else:
                    whole_df = whole_df.join(df, how='outer', on='Date')

            else:
                print(f"No data found for {ticker.ticker}, with period {self.period} and interval {self.interval}")

        return whole_df

    def get_ticker_data(self, ticker: str):
        """Load data for a specific ticker"""
        ticker_path = os.path.join(self.data_dir, f"{ticker}.csv")
        if os.path.exists(ticker_path):
            return pd.read_csv(ticker_path)
        return None

    def _validate_data_dir(self):
        """Create directory structure if it doesn't exist"""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def update_data(self, **kwargs):
        """Update existing data with new data"""
        for ticker in self.tickers:
            ticker_path = os.path.join(self.data_dir, f"{ticker.ticker}.csv")
            if os.path.exists(ticker_path):
                existing_data = pd.read_csv(ticker_path)
                existing_data['Date'] = pd.to_datetime(existing_data['Date'], errors='coerce')
                latest_date = existing_data['Date'].dropna().max()
                
                # Download new data
                new_data = ticker.history(start=latest_date, **kwargs)
                if not new_data.empty:
                    # Combine and save
                    updated_data = pd.concat([existing_data, new_data])
                    updated_data.to_csv(ticker_path, index=False)
                    print(f"Updated data for {ticker.ticker}")
            else:
                print(f"No existing data for {ticker.ticker}, downloading full history")
                self.download_data(**kwargs)

# %%
# Create instance
historical_data = HistoricalData(ALL_TICKERS, period="10y", interval="1wk")
historical_data.download_data()
# dta = historical_data.combine_ticker_data(['Close'])
# Load all data combined
# combined_data = historical_data.download_data()

