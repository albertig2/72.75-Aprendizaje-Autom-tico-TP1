import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def select_features(df):
    selected_features = ['age', 'bmi', 'smoker_yes', 'charges']
    return df[selected_features]

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

# Least correlation with the target variable: 'region_northwest', 'region_southeast', 'region_southwest', 'region_northeast', 'sex_male', 'sex_female', 'children'
# Should be exactly the same as the built-in pandas function:
# df['age'].corr(df['charges'])

def scatter_plot(df, col1, col2):
    plt.scatter(df[col1], df[col2])
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.title(f'Scatter Plot: {col1} vs {col2}')
    plt.show()

# Checking correlation between features to avoid multicollinearity
def check_multicollinearity(df):
    correlation_matrix = df.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()
