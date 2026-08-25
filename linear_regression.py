from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import numpy as np


def cross_validation_linear_regression(x_train_full, y_train_full, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    
    rmse = 0
    for fold, (train_idx, val_idx) in enumerate(kf.split(x_train_full)):
        # Split in train and validation sets
        x_train = x_train_full.iloc[train_idx]
        y_train = y_train_full.iloc[train_idx]
        x_val = x_train_full.iloc[val_idx]
        y_val = y_train_full.iloc[val_idx]
        
        # Train model
        model = linear_regression_model(x_train, y_train)
        
        # Evaluate on validation fold
        print(f"\nFold {fold + 1}:")
        rmse += evaluate_model(model, x_val, y_val) / k
    print(f"\nAverage RMSE across {k} folds: {rmse}")
    
def linear_regression_model(x_train, y_train):
    model = LinearRegression()
    model.fit(x_train, y_train)
    return model

def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    # Calculate the mean squared error
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    rmse = np.sqrt(mse)
    print(f"Root Mean Squared Error: {rmse}")
    return rmse
