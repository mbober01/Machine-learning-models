import numpy as np
import pandas as pd
import random
import math
from collections import Counter


class Utils:
    @staticmethod
    def normalize(df):
        for name in df.columns:
            if np.issubdtype(df[name].dtype, np.number):
                df[name] = (df[name] - df[name].min())/(df[name].max() - df[name].min())
        return df

    @staticmethod
    def split(df, size):
        train_size = int(size * len(df))
        train_df = df.iloc[:train_size]
        test_df = df.iloc[train_size:]
        return train_df, test_df

    @staticmethod
    def shuffle(df):
        dl = df.values.tolist()
        length = len(dl)
        for x in range(length):
            try:
                i = random.randint(0, length-x)
                dl[i], dl[-1-x] = dl[-1-x], dl[i]
            except:
                pass
        new_df = pd.DataFrame(data=dl, columns=df.columns)
        return new_df

    @staticmethod
    def distance(x, y, m=1):
        dist = (sum([pow(abs(a-b), m) for a, b in zip(x, y)])) ** (1/m)
        return dist



