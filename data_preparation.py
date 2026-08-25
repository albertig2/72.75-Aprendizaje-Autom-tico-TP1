import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression, f_regression
from data_preparation import prepare_data_set



def prepare_data_set(data_set):
    df_encoded = encode_categorical_variables(df)
    df_normalizado = normalize_z_score(df_encoded)
    return df_normalizado


# 1.1 Variables Categóricas
def encode_categorical_variables(df):
    categorical_columns = df.select_dtypes(include=['object']).columns
    encoder = OneHotEncoder()
    encoded_data = encoder.fit_transform(df[categorical_columns]).toarray()
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_columns))
    final_df = pd.concat([df.drop(columns=categorical_columns), encoded_df], axis=1 )
    return final_df


# Normalizar los datos para que todas las características tengan la misma escala y rango de valores. Esto es importante para evitar que algunas características dominen a otras en el proceso de entrenamiento del modelo.
def normalize_min_max_scale(df): # Not using at the moment
    return (df - df.min()) / (df.max() - df.min())

def normalize_z_score(df): # Better for the outliers, if we choose to keep the outliers
    return (df - df.mean()) / df.std()






