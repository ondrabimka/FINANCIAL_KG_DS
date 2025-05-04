import os
import pandas as pd

# Load all tickers from the CSV file from the environment variable DATA_PATH
ALL_TICKERS = list(pd.read_csv(os.getenv("DATA_PATH") + "/ticker_info.csv", usecols=["ticker"])["ticker"])