import matplotlib.pyplot as plt
from load_data import load_data_set

def smokers_histogram(df):
    smokers_df = df[df["smoker"].str.lower() == 'yes']
    nonsmokers_df = df[df["smoker"].str.lower() == 'no']
    
    plt.figure(figsize=(10, 6))
    plt.hist(smokers_df['charges'], bins=30, alpha=0.5, label='Smokers', color='red')
    plt.hist(nonsmokers_df['charges'], bins=30, alpha=0.5, label='Non-Smokers', color='blue')
    plt.title('Charges Histogram: Smokers vs Non-Smokers')
    plt.xlabel('Charges')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()
    
def bmi_scatter(df):
    smokers_df = df[df["smoker"].str.lower() == 'yes']
    nonsmokers_df = df[df["smoker"].str.lower() == 'no']
    
    plt.figure(figsize=(10, 6))
    plt.scatter(smokers_df['bmi'], smokers_df['charges'], alpha=0.5, label='Smokers', color='red')
    plt.scatter(nonsmokers_df['bmi'], nonsmokers_df['charges'], alpha=0.5, label='Non-Smokers', color='blue')
    plt.title('Charges vs BMI: Smokers vs Non-Smokers')
    plt.xlabel('BMI')
    plt.ylabel('Charges')
    plt.legend()
    plt.show()

    
if __name__ == "__main__":
    data_set = load_data_set()
    smokers_histogram(data_set)
    bmi_scatter(data_set)
