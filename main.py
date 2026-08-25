from load_data import load_data_set
from clean_data import clean_data_set
from data_preparation import prepare_data_set
from select_features import select_features
from linear_regression import cross_validation_linear_regression
from polynomial_regression import cross_validation_polynomial_regression

data_set = load_data_set()
data_set = clean_data_set(data_set)
data_set = prepare_data_set(data_set)
data_set = select_features(data_set, features)  # Select features based on correlation and domain knowledge

train = data_set.sample(frac=0.8, random_state=42)
test = data_set.drop(train.index)

x_train = train.drop("charges", axis=1)
y_train = train["charges"]
x_test = test.drop("charges", axis=1)
y_test = test["charges"]

cross_validation_linear_regression(x_train, y_train, k=5)
cross_validation_polynomial_regression(x_train, y_train, degree=2, k=5)


# Choosing characteriistics for the model based on correlation and domain knowledge

# Usamos filtros, no capturan interacciones entre features, pero son rápidos y fáciles de interpretar.

# Coeficientes muy grandes pueden indicar un modelo demasiado sensible a los datos
