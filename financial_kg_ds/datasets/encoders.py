# %%
import pandas as pd
import torch
from typing import Union

from financial_kg_ds.utils.cache import Cache
from financial_kg_ds.datasets.historical_data import HistoricalData
import matplotlib.pyplot as plt

class IdentityEncoder(object):
    """Converts list of floats to tensor."""

    def __init__(self, dtype=None):
        self.dtype = dtype

    def __call__(self, df):
        # scale the values to be between 0 and 1
        df = df.fillna(0)
        df = (df - df.min()) / (df.max() - df.min())
        return torch.from_numpy(df.values).view(-1, 1).to(self.dtype)


class OneHotEncoder(object):
    """Converts list of floats to tensor."""

    def __init__(self, dtype=None):
        self.dtype = dtype

    def __call__(self, df: pd.DataFrame):
        return torch.from_numpy(pd.get_dummies(df.drop_duplicates()).values)


from transformers import pipeline


class SentimentAnalysisEncoder(object):
    """Converts list of floats to tensor."""

    def __init__(self, dtype=None):
        self.dtype = dtype
        self.pipe = pipeline("text-classification", model="ProsusAI/finbert")
        self.cache = Cache("financial_kg_ds/data", "finbert_cache")

    def encode_news(self, title):
        if title in self.cache.cache:
            print(f"cache hit: {title}")
            print(self.cache.cache[title])
            return self.cache.cache[title]
        try:
            output = self.pipe(title)[0]
            # positive 1 negative -1 neutral 0
            if output["label"] == "positive":
                label = 1
            elif output["label"] == "negative":
                label = -1
            else:
                label = 0

            self.cache.cache[title] = label, output["score"]
            self.cache.save_cache()
            return label, output["score"]
        except Exception as e:
            print(f"Error: {e}: ", title)
            return 0, 0

    def __call__(self, df: pd.DataFrame):
        return torch.tensor([self.encode_news(title) for title in list(set(df.values))]).to(self.dtype)

# %%
from financial_kg_ds.models.BiRNN_autoencoder import LSTMAutoencoderBidi
from sklearn.preprocessing import StandardScaler
import torch
from tqdm import tqdm

class TimeSeriesEncoder(object):
    """
    Converts time series data to embeddings.
    
    Example
    -------
    encoder = TimeSeriesEncoder('financial_kg_ds/data/best_model_bidi_29_1_2025-01-08.pth')
    test_df = pd.read_csv('financial_kg_ds/data/historical_prices/historical_data_1h.csv', usecols=['Close_AAPL']
    encoder(test_df)
    """

    def __init__(self, period, interval, date_cut_off, window_size = 52, rnn_model_path: str = 'financial_kg_ds/data/best_model_bidi_21_1_2025-05-11_10y_1wk_2024-09-06.pth'):
        self.rnn_model = self._load_rnn_model(rnn_model_path)
        self.period = period
        self.interval = interval
        self.date_cut_off = date_cut_off
        self.window_size = window_size

    def _load_rnn_model(self, rnn_model_path: str):
        print("loading rnn model")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hidden_dim, num_layers = self._extract_params_from_model_path(rnn_model_path)
        model = LSTMAutoencoderBidi(1, hidden_dim, num_layers)
        model.load_state_dict(torch.load(rnn_model_path, map_location=device), strict=False)
        return model

    @staticmethod
    def _extract_params_from_model_path(rnn_model_path: str):
        hidden_dim = rnn_model_path.split('_')[5]
        num_layers = rnn_model_path.split('_')[6]
        return int(hidden_dim), int(num_layers)
        
    def _preprocess_df(self, df: pd.DataFrame, scaler=StandardScaler()):
        if len(df) < self.window_size:
            return torch.zeros((1, self.window_size, 1)).float()
        elif len(df) > self.window_size:
            df = df.tail(self.window_size)
            df = df.fillna(0)
            return torch.tensor(scaler.fit_transform(df)).reshape(-1,1).unsqueeze(0).float()
        
    def get_embedding(self, df: pd.DataFrame):
        input = self._preprocess_df(df)
        output = self.rnn_model.get_embedding(input)
        return output.squeeze().detach().numpy()

    def plot_input_vs_output(self, df: pd.DataFrame):
        input = self._preprocess_df(df)
        output = self.rnn_model(input)
        plt.plot(input.squeeze().detach().numpy())
        plt.plot(output.squeeze().detach().numpy())
        plt.show()

    def get_dataframe(self, tickers: Union[str, list, pd.DataFrame], columns: list = ['Close'], period: str = '10y', interval: str = '1wk'):

        if isinstance(tickers, pd.DataFrame):
            assert "ticker" in tickers.columns, "tickers should have a ticker column"
            tickers = tickers.ticker.to_list()
        elif isinstance(tickers, str):
            assert len(tickers) > 0, "tickers should not be empty"
            tickers = [tickers]
        assert isinstance(tickers, list), "tickers should be a list"

        historical_data = HistoricalData(tickers=tickers, period=period, interval=interval)
        historical_data_df = historical_data.combine_ticker_data(columns)
        historical_data_df = historical_data_df[historical_data_df.index < pd.to_datetime(self.date_cut_off).tz_localize('UTC')]
        return historical_data_df
            

    def __call__(self, df: pd.DataFrame, columns: list = ['Close'], period: str = '10y', interval: str = '1wk'):
        """
        df: pd.DataFrame
            Dataframe with historical data
        
        columns: list
            List of columns to use for the embedding
        """

        if isinstance(df, pd.Series):
            df = list(df)

        data = self.get_dataframe(df, columns, period, interval)
        print(f"data shape: {data.shape}")
        embeddings = []
        for col in tqdm(data.columns, desc="Encoding columns"):
            if col not in data.columns:
                print(f"column {col} not in data")
            embedding = self.get_embedding(data[[col]])
            embeddings.append(embedding)

        return torch.tensor(embeddings)# .T
    
    def plot_input_vs_output_for_ticker(self, ticker: str):
        """
        df: pd.DataFrame
            Dataframe with historical data
        
        columns: list
            List of columns to use for the embedding
        """

        data = self.get_dataframe(ticker, ['Close'], self.period, self.interval)
        self.plot_input_vs_output(data)



# %%
# tickers = ["AAPL", "MSFT", "GOOGL"]
# tickers_df = pd.DataFrame(tickers, columns=["ticker"])
# encoder = TimeSeriesEncoder("10y", "1wk", "2024-11-12")
# encoder.plot_input_vs_output_for_ticker("MSFT")
# encoder(tickers_df).shape
# %%
