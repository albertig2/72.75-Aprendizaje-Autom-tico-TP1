import matplotlib.pyplot as plt
import csv
from pathlib import Path
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


csv_path = Path(__file__).parent / 'data' / 'insurance.csv'
df = pd.read_csv(csv_path)

# 1.1 Variables Categóricas
def encode_categorical_variables(df):
    categorical_columns = df.select_dtypes(include=['object']).columns
    encoder = OneHotEncoder()
    encoded_data = encoder.fit_transform(df[categorical_columns]).toarray()
    encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(categorical_columns))
    final_df = pd.concat([df.drop(columns=categorical_columns), encoded_df], axis=1 )
    return final_df

encoded_df = encode_categorical_variables(df)

# 1.2 Valores Faltantes
def report_missing_values(df):
    missing_values = df.isnull().sum()
    print("Missing Values Report:")
    for column, count in missing_values.items():
        if count > 0:
            print(f"{column}: {count} missing values")
            
report_missing_values(df)

def drop_row_of_missing_value(df):
    df_clean = df.dropna()
    return df_clean

def drop_column_of_missing_value(df):
    df_clean = df.dropna(axis=1)
    return df_clean

def save_cleaned_data(df_clean, output_path):
    df_clean.to_csv(output_path, index=False)
    
# 1.3 Outliers
# def detect_outliers(df, column):
def plot_boxplots(df):
    age = df['age']
    sex = df['sex']
    bmi = df['bmi']
    children = df['children']
    smoker = df['smoker']
    region = df['region']
    charges = df['charges']
    
    figure = plt.figure(figsize=(10, 6))
    plt.boxplot(age, vert=False)
    plt.title('Age Boxplot')
    plt.show()

    figure = plt.figure(figsize=(10, 6))
    plt.boxplot(bmi, vert=False)
    plt.title('BMI Boxplot')
    plt.show()

    figure = plt.figure(figsize=(10, 6))
    plt.boxplot(children, vert=False)
    plt.title('Children Boxplot')
    plt.show()

    figure = plt.figure(figsize=(10, 6))
    plt.boxplot(charges, vert=False)
    plt.title('Charges Boxplot')
    plt.show()

def plot_histograms(df):
    age = df['age']
    sex = df['sex']
    bmi = df['bmi']
    children = df['children']
    smoker = df['smoker']
    region = df['region']
    charges = df['charges']

    figure = plt.figure(figsize=(10, 6))
    plt.hist(age, bins=140)
    plt.title('Age Histogram')
    plt.show()

    figure = plt.figure(figsize=(10, 6))
    plt.hist(bmi, bins=30)
    plt.title('BMI Histogram')
    plt.show()

    figure = plt.figure(figsize=(10, 6))
    plt.hist(children, bins=10)
    plt.title('Children Histogram')
    plt.show()

    figure = plt.figure(figsize=(10, 6))
    plt.hist(charges, bins=30)
    plt.title('Charges Histogram')
    plt.show()





# Choosing characteriistics for the model based on correlation and domain knowledge

# Normalizar los datos para que todas las características tengan la misma escala y rango de valores. Esto es importante para evitar que algunas características dominen a otras en el proceso de entrenamiento del modelo.
def normalize_min_max_scale(df):
    return (df - df.min()) / (df.max() - df.min())

def normalize_z_score(df):
    return (df - df.mean()) / df.std()

df_normalizado_mm_scale = normalize_min_max_scale(encoded_df) # Muy sensible a outliers, pero mantiene la distribución de los datos.
df_normalizado_z_score = normalize_z_score(encoded_df) # Menos sensible a outliers, pero puede distorsionar la distribución de los datos.

print(df_normalizado_mm_scale.head())

data_set = df_normalizado_z_score.drop(columns=['region_northwest', 'region_southeast', 'region_southwest', 'region_northeast', 'sex_male', 'sex_female'])
# 1.4 Características
# Pearson
def pearson_correlation(df, col1, col2):
    cov_matrix = np.cov(df[col1], df[col2])
    cov_xy = cov_matrix[0, 1]
    std_x = np.std(df[col1])
    std_y = np.std(df[col2])
    correlation = cov_xy / (std_x * std_y)
    
    correlation = df[col1].corr(df[col2])
    return correlation
"""
print(pearson_correlation(encoded_df, 'age', 'charges'))
print(pearson_correlation(encoded_df, 'bmi', 'charges'))
print(pearson_correlation(encoded_df, 'children', 'charges'))
print(pearson_correlation(encoded_df, 'smoker_yes', 'charges'))
print(pearson_correlation(encoded_df, 'region_northwest', 'charges'))
print(pearson_correlation(encoded_df, 'region_southeast', 'charges'))
print(pearson_correlation(encoded_df, 'region_southwest', 'charges'))
print(pearson_correlation(encoded_df, 'region_northeast', 'charges'))
print(pearson_correlation(encoded_df, 'sex_male', 'charges'))
"""

# Least correlation with the target variable: 'region_northwest', 'region_southeast', 'region_southwest', 'region_northeast', 'sex_male', 'sex_female', 'children'
# Should be exactly the same as the built-in pandas function:
# df['age'].corr(df['charges'])

def scatter_plot(df, col1, col2):
    plt.scatter(df[col1], df[col2])
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.title(f'Scatter Plot: {col1} vs {col2}')
    plt.show()
    
# selecting features based on correlation and domain knowledge

scatter_plot(encoded_df, 'age', 'charges')
scatter_plot(encoded_df, 'bmi', 'charges')
scatter_plot(encoded_df, 'children', 'charges')
scatter_plot(encoded_df, 'smoker_yes', 'charges')
scatter_plot(encoded_df, 'region_northwest', 'charges')
scatter_plot(encoded_df, 'region_southeast', 'charges')
scatter_plot(encoded_df, 'region_southwest', 'charges')
scatter_plot(encoded_df, 'region_northeast', 'charges')
scatter_plot(encoded_df, 'sex_male', 'charges')



# Checking correlation between features to avoid multicollinearity
def check_multicollinearity(df):
    correlation_matrix = df.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()

check_multicollinearity(encoded_df)

# Have to use age, bmi, and smoking
# So far just getting linear relations, 


# but we can also check for non-linear relations using scatter plots.
# I think that number of children seemed to maybe have a non-linear relation with charges.....

# Checking nonlineal relations: 

# Features and target

X = encoded_df.drop("charges", axis=1)

y = encoded_df["charges"]

# Compute mutual information

mi_scores = mutual_info_regression(X, y, random_state=42)

mi_df = pd.DataFrame({"Feature": X.columns, "MI Score": mi_scores}).sort_values("MI Score", ascending=False)
print(mi_df)

mi_df = mi_df.sort_values("MI Score")

plt.figure(figsize=(8,5))
sns.barplot(data=mi_df, x="MI Score", y="Feature",palette="viridis")
plt.title("Mutual Information")
plt.xlabel("MI Score")
plt.ylabel("Feature")
plt.show()

## ANOVA F-test
f_scores, p_values = f_regression(X, y)

results = pd.DataFrame({
    "Feature": X.columns,
    "F-score": f_scores,
    "p-value": p_values
}).sort_values("F-score", ascending=False)

print(results)

# Usamos filtros, no capturan interacciones entre features, pero son rápidos y fáciles de interpretar.



# REGRESSION MODELS:
# Coeficientes muy grandes pueden indicar un modelo demasiado sensible a los datos








# METRICAS para medir

# REgression linear: 
# Validacion cruzada: K-fold cross-validation

df_normalizado = normalize_z_score(encoded_df)
train = df_normalizado.sample(frac=0.8, random_state=42)
test = df_normalizado.drop(train.index)

#Should drop the features we dont care about here 
train_x = train.drop("charges", axis=1)
train_y = train["charges"]
test_x = test.drop("charges", axis=1)
test_y = test["charges"]

model = LinearRegression()
model.fit(train_x, train_y)
y_pred = model.predict(test_x)

# Calculate the mean squared error
mse = mean_squared_error(test_y, y_pred)
print(f"Mean Squared Error: {mse}")
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error: {rmse}")

#e = y_pred - test_y

# Regression polynomial:



