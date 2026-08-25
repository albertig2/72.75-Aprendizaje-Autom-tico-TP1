from load_data import load_data_set
from clean_data import clean_data_set
from data_preparation import prepare_data_set
from select_features import select_features
from regression import cross_validation_regression, evaluate_model

data_raw = load_data_set()
data_cleaned = clean_data_set(data_raw)
data_prepared = prepare_data_set(data_cleaned)
data_set = data_prepared
data_set = select_features(data_set, 'all')  # Select features based on correlation and domain knowledge

train = data_set.sample(frac=0.8, random_state=42)
test = data_set.drop(train.index)

x_train = train.drop("charges", axis=1)
y_train = train["charges"]
x_test = test.drop("charges", axis=1)
y_test = test["charges"]

# Linear regression with cross-validation
model1, poly1 = cross_validation_regression(x_train, y_train, degree=1, k=5)
# Polynomial regression with cross-validation
model2, poly2 = cross_validation_regression(x_train, y_train, degree=2, k=5)
model3, poly3 = cross_validation_regression(x_train, y_train, degree=3, k=5)

evaluate_model(model1, poly1, x_test, y_test)
