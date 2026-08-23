"""
TP1 - Regression and Introduction to Model Evaluation
Step 3: Polynomial regression, and step 4: evaluation.

3.1 Polynomial transformation of the input variables
3.2 Linear regression trained on the transformed variables
3.3 L1 (Lasso) regularisation with several values of lambda
4.  Validation RMSE reported for every degree and every lambda

Everything reuses the split and the k-fold scheme from linear_regression.py,
so all models are compared on exactly the same folds. Any difference in RMSE
is therefore caused by the model, not by a different partition of the data.
"""

import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from data_prep import prepare
from linear_regression import (N_SPLITS, RANDOM_STATE, cross_validate_rmse,
                               split_data)

# --- Configuration -----------------------------------------------------------

DEGREES = [1, 2, 3]
LAMBDAS = [1.0, 10.0, 100.0, 1000.0]

# In scikit-learn the L1 strength is called alpha; it is the lambda of the
# course notes. Written out explicitly to avoid confusion in the report.


# --- 3.1 Polynomial pipeline -------------------------------------------------

def build_poly_pipeline(numeric_columns, model=None, degree=2):
    """
    Standardise -> expand polynomially -> standardise again -> fit.

    Step order matters:

    1. Scale the numeric columns first. Expanding raw values would produce
       terms such as bmi**3 on the order of 10**5, which conditions the
       design matrix badly.
    2. PolynomialFeatures with include_bias=False, since the model already
       fits its own intercept. interaction_only=False, so both powers
       (age**2) and interactions (bmi * smoker_yes) are generated. The
       interactions are the interesting part here: the effect of bmi on
       charges is much stronger for smokers, and a purely additive model
       cannot express that.
    3. Scale again. Even starting from standardised inputs, a squared term
       has a different spread from a linear one. Lasso penalises every
       coefficient with the same lambda, so without this step the penalty
       would fall unevenly across terms purely because of their scale.
    """
    if model is None:
        model = LinearRegression()

    preprocessor = ColumnTransformer(
        transformers=[("scale", StandardScaler(), numeric_columns)],
        remainder="passthrough",
    )
    return Pipeline([
        ("preprocess", preprocessor),
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("rescale", StandardScaler()),
        ("model", model),
    ])


def poly_factory(degree):
    """Adapt build_poly_pipeline to the (numeric_columns, model) signature
    that cross_validate_rmse expects."""
    def factory(numeric_columns, model):
        return build_poly_pipeline(numeric_columns, model, degree=degree)
    return factory


def count_features(X, numeric_columns, degree):
    """How many features the expansion produces, for the report."""
    pipeline = build_poly_pipeline(numeric_columns, degree=degree)
    pipeline.fit(X.head(50), np.zeros(50))
    return pipeline.named_steps["poly"].n_output_features_


# --- 3.2 Unregularised polynomial models -------------------------------------

def evaluate_degrees(X_train, y_train, numeric_columns, degrees=DEGREES,
                     n_splits=N_SPLITS):
    """Cross-validate plain linear regression at each polynomial degree."""
    print("\n" + "-" * 68)
    print("3.1 / 3.2  POLYNOMIAL EXPANSION, NO REGULARISATION")
    print("-" * 68)

    rows = []
    for degree in degrees:
        n_features = count_features(X_train, numeric_columns, degree)
        results = cross_validate_rmse(
            X_train, y_train, numeric_columns,
            model=LinearRegression(),
            n_splits=n_splits,
            pipeline_factory=poly_factory(degree),
            verbose=False,
        )
        rows.append({
            "degree": degree,
            "lambda": 0.0,
            "n_features": n_features,
            "rmse_train": results["rmse_train"].mean(),
            "rmse_val": results["rmse_val"].mean(),
            "val_sd": results["rmse_val"].std(),
            "gap": results["rmse_val"].mean() - results["rmse_train"].mean(),
        })
        print("  degree %d: %3d features -> train %8.2f | val %8.2f | gap %8.2f"
              % (degree, n_features, rows[-1]["rmse_train"],
                 rows[-1]["rmse_val"], rows[-1]["gap"]))
    return pd.DataFrame(rows)


# --- 3.3 L1 regularisation ---------------------------------------------------

def evaluate_lasso(X_train, y_train, numeric_columns, degrees=DEGREES,
                   lambdas=LAMBDAS, n_splits=N_SPLITS):
    """Cross-validate Lasso at each (degree, lambda) combination."""
    print("\n" + "-" * 68)
    print("3.3  L1 REGULARISATION (Lasso). sklearn 'alpha' = lambda")
    print("-" * 68)

    rows = []
    for degree in degrees:
        n_features = count_features(X_train, numeric_columns, degree)
        for lam in lambdas:
            model = Lasso(alpha=lam, max_iter=100000, tol=1e-3,
                          random_state=RANDOM_STATE)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                results = cross_validate_rmse(
                    X_train, y_train, numeric_columns,
                    model=model,
                    n_splits=n_splits,
                    pipeline_factory=poly_factory(degree),
                    verbose=False,
                )
            rows.append({
                "degree": degree,
                "lambda": lam,
                "n_features": n_features,
                "rmse_train": results["rmse_train"].mean(),
                "rmse_val": results["rmse_val"].mean(),
                "val_sd": results["rmse_val"].std(),
                "gap": (results["rmse_val"].mean()
                        - results["rmse_train"].mean()),
            })
            print("  degree %d, lambda %7.1f -> train %8.2f | val %8.2f | "
                  "gap %8.2f" % (degree, lam, rows[-1]["rmse_train"],
                                 rows[-1]["rmse_val"], rows[-1]["gap"]))
    return pd.DataFrame(rows)


def count_zeroed_coefficients(X_train, y_train, numeric_columns, degree, lam):
    """
    How many coefficients Lasso drives to exactly zero.

    This is the practical argument for L1 over L2: it performs feature
    selection, not just shrinkage.
    """
    model = Lasso(alpha=lam, max_iter=100000, tol=1e-3,
                  random_state=RANDOM_STATE)
    pipeline = build_poly_pipeline(numeric_columns, model, degree=degree)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        pipeline.fit(X_train, y_train)
    coefs = pipeline.named_steps["model"].coef_
    n_zero = int(np.sum(coefs == 0))
    return n_zero, len(coefs)


# --- 4. Evaluation summary ---------------------------------------------------

def report_summary(all_results, X_train, y_train, numeric_columns):
    """Section 4: full table of validation RMSE per degree and lambda."""
    print("\n" + "=" * 68)
    print("4.  EVALUATION - validation RMSE by degree and lambda")
    print("=" * 68)

    table = all_results.copy()
    table["model"] = np.where(table["lambda"] == 0, "OLS", "Lasso")
    display = table[["model", "degree", "lambda", "n_features",
                     "rmse_train", "rmse_val", "gap"]]
    print(display.round(2).to_string(index=False))

    best = table.loc[table["rmse_val"].idxmin()]
    print("\nLowest validation RMSE:")
    print("  model  : %s" % best["model"])
    print("  degree : %d" % best["degree"])
    print("  lambda : %.1f" % best["lambda"])
    print("  val RMSE %.2f  (train %.2f, gap %.2f)"
          % (best["rmse_val"], best["rmse_train"], best["gap"]))

    if best["model"] == "Lasso":
        n_zero, n_total = count_zeroed_coefficients(
            X_train, y_train, numeric_columns,
            int(best["degree"]), float(best["lambda"]))
        print("  L1 set %d of %d coefficients to exactly zero (%.0f%%)."
              % (n_zero, n_total, 100 * n_zero / n_total))

    return best


# --- Entry point -------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="TP1 - polynomial regression and evaluation")
    parser.add_argument("--data", default="insurance.csv")
    parser.add_argument("--folds", type=int, default=N_SPLITS)
    args = parser.parse_args()

    X, y, numeric_columns = prepare(args.data)
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("=" * 68)
    print("SECTIONS 3 AND 4 - POLYNOMIAL REGRESSION")
    print("=" * 68)
    print("Same split and same folds as section 2:")
    print("  train %d rows, test %d rows (still untouched)"
          % (len(X_train), len(X_test)))
    print("  %d-fold CV, seed %d" % (args.folds, RANDOM_STATE))

    poly_results = evaluate_degrees(X_train, y_train, numeric_columns,
                                    n_splits=args.folds)
    lasso_results = evaluate_lasso(X_train, y_train, numeric_columns,
                                   n_splits=args.folds)

    all_results = pd.concat([poly_results, lasso_results], ignore_index=True)
    report_summary(all_results, X_train, y_train, numeric_columns)

    all_results.to_csv("results_polynomial.csv", index=False)
    print("\nFull results written to results_polynomial.csv")
    print("The test set has still not been evaluated.")


if __name__ == "__main__":
    main()
