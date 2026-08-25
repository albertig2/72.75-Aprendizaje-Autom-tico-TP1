import matplotlib.pyplot as plt
from load_data import load_data_set


def clean_data_set(df):
    df_clean = df.dropna()
    df_clean = df.drop_duplicates()
    return df_clean


def report_data(data_set):
    report_missing_values(data_set)
    report_duplicates(data_set)


def report_missing_values(df):
    missing_values = df.isnull().sum()
    print("Missing Values Report:")
    if missing_values.sum() == 0:
            print("No missing values found.")
    for column, count in missing_values.items():
        if count > 0:
            print(f"{column}: {count} missing values")
    

def report_duplicates(df):
    duplicate_count = df.duplicated().sum()
    print(f"Duplicate Rows Report: {duplicate_count} duplicate rows found")


if __name__ == "__main__":
    data_set = load_data_set()
    report_data(data_set)