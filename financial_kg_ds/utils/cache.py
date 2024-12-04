import os
import pickle


class Cache:
    def __init__(self, cache_dir: str, cache_name: str):
        self.cache_dir = cache_dir
        self.cache_name = cache_name
        self.cache_path = os.path.join(self.cache_dir, self.cache_name)
        self.cache = self.load_cache()

    def load_cache(self):
        if os.path.exists(self.cache_path):
            with open(self.cache_path, "rb") as f:
                return pickle.load(f)
        return {}

    def save_cache(self):
        with open(self.cache_path, "wb") as f:
            pickle.dump(self.cache, f)

    def get(self, key):
        return self.cache.get(key)

    def set(self, key, value):
        self.cache[key] = value
        self.save_cache()

    def delete(self, key):
        self.cache.pop(key, None)
        self.save_cache()

    def clear(self):
        self.cache = {}
        self.save_cache()
