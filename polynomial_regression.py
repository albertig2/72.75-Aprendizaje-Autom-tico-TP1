from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np

def cross_validation_polynomial_regression(x_train_full, y_train_full, degree, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    
    rmse = 0
    for fold, (train_idx, val_idx) in enumerate(kf.split(x_train_full)):
        # Split in train and validation sets
        x_train = x_train_full.iloc[train_idx]
        y_train = y_train_full.iloc[train_idx]
        x_val = x_train_full.iloc[val_idx]
        y_val = y_train_full.iloc[val_idx]
        
        # Train model
        model = polynomial_regression_model(2, x_train, y_train)
        
        # Evaluate on validation fold
        print(f"\nFold {fold + 1}:")
        rmse += evaluate_model(model, degree,x_val, y_val) / k
    print(f"\nAverage RMSE across {k} folds: {rmse}")

def polynomial_regression_model(degree, x_train, y_train):
    poly = PolynomialFeatures(degree=degree)
    x_train_poly = poly.fit_transform(x_train)
    model = LinearRegression()
    model.fit(x_train_poly, y_train)
    return model

def evaluate_model(model, degree, x_test, y_test):
    poly = PolynomialFeatures(degree=degree)
    x_test_poly = poly.transform(x_test)  
    y_pred = model.predict(x_test_poly)
    # Calculate the mean squared error
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")
    rmse = np.sqrt(mse)
    print(f"Root Mean Squared Error: {rmse}")
    return rmse
    