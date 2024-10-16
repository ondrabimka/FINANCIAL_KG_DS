import torch
import pandas as pd
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

    def __call__(self, df:pd.DataFrame):
        return torch.from_numpy(pd.get_dummies(df.drop_duplicates()).values)
    
from transformers import pipeline

class SentimentAnalysisEncoder(object):
    """Converts list of floats to tensor."""
    def __init__(self, dtype=None):
        self.dtype = dtype
        self.pipe = pipeline("text-classification", model="ProsusAI/finbert")
        self.cache = Cache('data', 'finbert_cache')

    def encode_news(self, title):

        if title in self.cache.cache:
            print(f'cache hit: {title}')
            print(self.cache.cache[title])
            return self.cache.cache[title]
        try:
            output = self.pipe(title)[0]
            # positive 1 negative -1 neutral 0
            if output['label'] == 'positive':
                label = 1
            elif output['label'] == 'negative':
                label = -1
            else:
                label = 0

            self.cache.cache[title] = label, output['score']
            self.cache.save_cache()
            return label, output['score']
        except:
            return 0, 0

    def __call__(self, df:pd.DataFrame):
        return torch.tensor([self.encode_news(title) for title in df.values]).to(self.dtype)