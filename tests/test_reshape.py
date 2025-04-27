import sys
import unittest

import numpy as np
import pandas as pd

sys.path.append("..")
from financial_kg_ds.datasets.rnn_loader import RNNLoader


class TestRNNLoader(unittest.TestCase):
    def test_ae_from_dataframe(self):
        data = {
            "Close_AAPL": [1, 5, 9, 13, 17],
            "Volume_AAPL": [2, 6, 10, 14, 18],
            "Close_AMZN": [3, 7, 11, 15, 19],
            "Volume_AMZN": [4, 8, 12, 16, 20],
        }
        df = pd.DataFrame(data)
        window_size = 3
        batch_size = 5
        shuffle = False

        loader = RNNLoader.ae_from_dataframe(df, window_size, batch_size, shuffle, scaler=None)
        X, _ = next(iter(loader))

        expected_X = np.array(
            [
                [[1, 2], [5, 6], [9, 10]],
                [[5, 6], [9, 10], [13, 14]],
                [[9, 10], [13, 14], [17, 18]],
                [[3, 4], [7, 8], [11, 12]],
                [[7, 8], [11, 12], [15, 16]],
            ]
        )
        np.testing.assert_array_equal(X, expected_X)


if __name__ == "__main__":
    unittest.main()
