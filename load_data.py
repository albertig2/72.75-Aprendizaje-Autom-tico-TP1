import pandas as pd


def load_data_set(path="insurance.csv"):
    return pd.read_csv(path)
