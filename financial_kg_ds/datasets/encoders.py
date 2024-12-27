import pandas as pd
import torch

from financial_kg_ds.utils.cache import Cache


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
        self.cache = Cache("data", "finbert_cache")

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
        except:
            return 0, 0

    def __call__(self, df: pd.DataFrame):
        return torch.tensor([self.encode_news(title) for title in df.values]).to(self.dtype)
    
from financial_kg_ds.models.RNN_autoencoder import LSTMAutoencoderBidi
from sklearn.preprocessing import MinMaxScaler

    """Converts time series data to embeddings."""

    def __init__(self, rnn_model_path: str):
        self.rnn_model = self._load_rnn_model(rnn_model_path)

    def _load_rnn_model(self, rnn_model_path: str):
        print("loading rnn model")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        hidden_dim, num_layers = self._extract_params_from_model_path(rnn_model_path)
        model = LSTMAutoencoderBidi(2, 1, num_layers)
        model.load_state_dict(torch.load(rnn_model_path, map_location=device), strict=False)
        return model

    @staticmethod
    def _extract_params_from_model_path(rnn_model_path: str):
        hidden_dim = rnn_model_path.split('_')[5]
        num_layers = rnn_model_path.split('_')[6]
        print("hidden_dim: ", hidden_dim, "num_layers: ", num_layers)
        return int(hidden_dim), int(num_layers)
        
    def _preprocess_df(self, df, window_size=49, scaler=MinMaxScaler()):
        if len(df) < window_size:
            raise ValueError("Dataframe is too small")
        elif len(df) > window_size:
            df = df.tail(window_size)
        df = df.fillna(0)
        return scaler.fit_transform(df)

    def __call__(self, df):
        df = self._preprocess_df(df)
        return self.rnn_model.get_embedding(df)