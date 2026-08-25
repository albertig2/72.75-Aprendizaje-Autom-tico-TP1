import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from load_data import load_data_set

def prepare_data_set(data_set):
    df_encoded = encode_categorical_variables(data_set)
    df_normalized = normalize_z_score(df_encoded)
    return df_normalized


def encode_categorical_variables(df):
    df_encoded = df.copy()
    df_encoded["sex"] = df_encoded["sex"].map({"male": 0, "female": 1})
    df_encoded["smoker"] = df_encoded["smoker"].map({"no": 0, "yes": 1})

    categorical_columns = df_encoded.select_dtypes(include=["object", "string", "str"]).columns
    if len(categorical_columns) == 0:
        return df_encoded

    encoder = OneHotEncoder()
    encoded_data = encoder.fit_transform(df_encoded[categorical_columns]).toarray()
    encoded_df = pd.DataFrame(
        encoded_data,
        columns=encoder.get_feature_names_out(categorical_columns),
        index=df_encoded.index,
    )
    final_df = pd.concat([df_encoded.drop(columns=categorical_columns), encoded_df], axis=1)
    return final_df


def normalize_z_score(df):
    return (df - df.mean()) / df.std()






