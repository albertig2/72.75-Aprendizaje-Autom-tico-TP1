import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression, f_regression
from load_data import load_data_set
from clean_data import clean_data_set
from data_preparation import prepare_data_set

def select_features(df, features):
    if features == "all":
        return df
    selected_features = features + ["charges"]
    print(df[selected_features].head())
    return df[selected_features]

def correlation_matrix_plot(df):
    correlation_matrix = df.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()

def mutual_information_plot(df):
    features = df.drop(columns=["charges"])
    target = df["charges"]
    mi_scores = mutual_info_regression(features, target, random_state=42)
    mi_df = pd.DataFrame({"Feature": features.columns, "MI Score": mi_scores}).sort_values("MI Score", ascending=False)
    plt.figure(figsize=(8, 5))
    sns.barplot(data=mi_df, x="MI Score", y="Feature", palette="viridis")
    plt.title("Mutual Information")
    plt.xlabel("MI Score")
    plt.ylabel("Feature")
    plt.show()

def anova_f_test(df):
    features = df.drop(columns=["charges"])
    target = df["charges"]
    f_scores, p_values = f_regression(features, target)

    results = pd.DataFrame({
        "Feature": features.columns,
        "F-score": f_scores,
        "p-value": p_values
    }).sort_values("F-score", ascending=False)

    print(results)


if __name__ == "__main__":
    data_raw = load_data_set()
    data_cleaned = clean_data_set(data_raw)
    data_prepared = prepare_data_set(data_cleaned)
    mutual_information_plot(data_prepared)
    anova_f_test(data_prepared)
