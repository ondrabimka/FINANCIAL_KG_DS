from json import load
import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Load all tickers from the CSV file from the environment variable DATA_PATH
ALL_TICKERS = list(pd.read_csv(os.getenv("TRAIN_DATA_PATH") + "/ticker_info.csv", usecols=["ticker"])["ticker"])