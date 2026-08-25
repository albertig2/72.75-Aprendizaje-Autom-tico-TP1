import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def prepare_data_set(data_set):
    df_encoded = encode_categorical_variables(data_set)
    df_normalize = normalize_z_score(df_encoded)
    return df_normalize


# 1.1 Variables Categóricas
def encode_categorical_variables(df):
    categorical_columns = df.select_dtypes(include=['object', 'str']).columns
    encoder = OneHotEncoder()
    encoded_data = encoder.fit_transform(df[categorical_columns]).toarray() # All categories are turned into a binary indicator column. fit transform encodes them and toarray converts it to a NumPy array
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_columns)) # Turns them back into readable names
    final_df = pd.concat([df.drop(columns=categorical_columns), encoded_df], axis=1 )
    return final_df # Result: categories are replaced by one-hot equivalents


# Normalizar los datos para que todas las características tengan la misma escala y rango de valores. Esto es importante para evitar que algunas características dominen a otras en el proceso de entrenamiento del modelo.
def normalize_min_max_scale(df): # Not using at the moment
    return (df - df.min()) / (df.max() - df.min())

def normalize_z_score(df): # Better for the outliers, if we choose to keep the outliers
    return (df - df.mean()) / df.std()






