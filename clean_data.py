import matplotlib.pyplot as plt


def clean_data_set(df):
    return drop_row_of_missing_value(df)


def evaluate_data(data_set):
    # Report missing values
    report_missing_values(data_set)
    
    # Plot boxplots for outlier detection
    plot_boxplots(data_set)
    
    # Plot histograms for distribution analysis
    plot_histograms(data_set)

# 1.2 Valores Faltantes
def report_missing_values(df):
    missing_values = df.isnull().sum()
    print("Missing Values Report:")
    for column, count in missing_values.items():
        if count > 0:
            print(f"{column}: {count} missing values")
    
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


# If wanting to remove data:

def drop_row_of_missing_value(df):
    df_clean = df.dropna()
    return df_clean

def drop_column_of_missing_value(df):
    df_clean = df.dropna(axis=1)
    return df_clean

def save_cleaned_data(df_clean, output_path):
    df_clean.to_csv(output_path, index=False)