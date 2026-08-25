import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression, f_regression


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


# Exploratory analysis on the encoded (but not yet feature-selected) DataFrame.
# Not called from main.py -- run manually, e.g. from a notebook or REPL.
def analyze_features(df):
    for col in ['age', 'bmi', 'children', 'smoker_yes', 'region_northwest',
                'region_southeast', 'region_southwest', 'region_northeast', 'sex_male']:
        print(pearson_correlation(df, col, 'charges'))

    for col in ['age', 'bmi', 'children', 'smoker_yes', 'region_northwest',
                'region_southeast', 'region_southwest', 'region_northeast', 'sex_male']:
        scatter_plot(df, col, 'charges')

    check_multicollinearity(df)

    # Have to use age, bmi, and smoking
    # So far just getting linear relations,

    # but we can also check for non-linear relations using scatter plots.
    # I think that number of children seemed to maybe have a non-linear relation with charges.....

    # Checking nonlineal relations:

    # Features and target
    X = df.drop("charges", axis=1)
    y = df["charges"]

    # Compute mutual information
    mi_scores = mutual_info_regression(X, y, random_state=42)

    mi_df = pd.DataFrame({"Feature": X.columns, "MI Score": mi_scores}).sort_values("MI Score", ascending=False)
    print(mi_df)

    mi_df = mi_df.sort_values("MI Score")

    plt.figure(figsize=(8, 5))
    sns.barplot(data=mi_df, x="MI Score", y="Feature", palette="viridis")
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
