"""
TP1 - Regression and Introduction to Model Evaluation
Step 2: Linear regression with k-fold cross-validation.

2.1 Train/test split
2.2 k-fold cross-validation on the TRAINING set only
2.3 Linear regression trained inside each fold, reporting train and
    validation RMSE

The test set is created here but never touched. It stays untouched until
section 5, when the selected model is evaluated once. Looking at it earlier
would turn it into a second validation set and the final RMSE estimate would
no longer be honest.
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data_prep import prepare

# --- Configuration -----------------------------------------------------------

TEST_SIZE = 0.20
N_SPLITS = 5
RANDOM_STATE = 42  # fixed so every run is reproducible


# --- 2.1 Train / test split --------------------------------------------------

def split_data(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """
    Hold out a test set with a simple random split.

    A random split is appropriate here: the rows are independent policy
    holders with no time ordering and no grouping, so a random sample is
    representative of the whole. (A time series such as the bike sharing
    dataset would require a chronological split instead.)

    Stratification is not used because the target is continuous.
    """
    return train_test_split(X, y, test_size=test_size,
                            random_state=random_state)


# --- Model pipeline ----------------------------------------------------------

def build_pipeline(numeric_columns, model=None):
    """
    Scaler + model as a single estimator.

    Wrapping both in a Pipeline is what makes the cross-validation honest:
    when the pipeline is fitted on a fold's training portion, the scaler
    computes its mean and standard deviation from that portion only. The
    validation portion is transformed with those values but never
    contributes to them.

    Only the genuinely numeric columns are standardised; the 0/1 dummies
    are passed through unchanged, since scaling them would destroy their
    interpretation without any numerical benefit.
    """
    if model is None:
        model = LinearRegression()

    preprocessor = ColumnTransformer(
        transformers=[("scale", StandardScaler(), numeric_columns)],
        remainder="passthrough",
    )
    return Pipeline([("preprocess", preprocessor), ("model", model)])


# --- 2.2 + 2.3 Cross-validation ----------------------------------------------

def cross_validate_rmse(X_train, y_train, numeric_columns, model=None,
                        n_splits=N_SPLITS, random_state=RANDOM_STATE,
                        label="Linear regression", pipeline_factory=None,
                        verbose=True):
    """
    Run k-fold CV on the training set and return per-fold RMSE.

    For each fold the pipeline is rebuilt from scratch, so no information
    is carried over between folds.

    pipeline_factory lets section 3 plug in a polynomial pipeline while
    reusing this exact function, so the folds and the split are guaranteed
    to be identical across all models being compared. It must be a callable
    taking (numeric_columns, model) and returning an unfitted estimator.

    Returns a DataFrame with one row per fold plus the mean and standard
    deviation of the train and validation RMSE.
    """
    if pipeline_factory is None:
        pipeline_factory = build_pipeline

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    records = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), start=1):
        X_fold_train = X_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]
        y_fold_train = y_train[train_idx]
        y_fold_val = y_train[val_idx]

        pipeline = pipeline_factory(numeric_columns, model)
        pipeline.fit(X_fold_train, y_fold_train)

        records.append({
            "fold": fold,
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "rmse_train": root_mean_squared_error(
                y_fold_train, pipeline.predict(X_fold_train)),
            "rmse_val": root_mean_squared_error(
                y_fold_val, pipeline.predict(X_fold_val)),
        })

    results = pd.DataFrame(records)
    if verbose:
        print("\n%s - %d-fold cross-validation" % (label, n_splits))
        print(results.round(2).to_string(index=False))
        print("  mean train RMSE: %9.2f  (sd %.2f)"
              % (results["rmse_train"].mean(), results["rmse_train"].std()))
        print("  mean val   RMSE: %9.2f  (sd %.2f)"
              % (results["rmse_val"].mean(), results["rmse_val"].std()))
        print("  gap (val - train): %7.2f"
              % (results["rmse_val"].mean() - results["rmse_train"].mean()))
    return results


# --- Interpretation helpers --------------------------------------------------

def show_coefficients(X_train, y_train, numeric_columns):
    """
    Fit once on the whole training set to inspect the coefficients.

    This is for interpretation only, not for scoring. Because the numeric
    features are standardised, their coefficients are in units of one
    standard deviation and can be compared with each other.
    """
    pipeline = build_pipeline(numeric_columns)
    pipeline.fit(X_train, y_train)

    feature_names = pipeline.named_steps["preprocess"].get_feature_names_out()
    coefs = pipeline.named_steps["model"].coef_
    table = (pd.DataFrame({"feature": [n.split("__")[-1] for n in feature_names],
                           "coefficient": coefs})
             .sort_values("coefficient", key=abs, ascending=False))

    print("\nCoefficients (fitted on the full training set, for interpretation)")
    print(table.round(2).to_string(index=False))
    print("  intercept: %.2f" % pipeline.named_steps["model"].intercept_)
    return table


def baseline_rmse(y_train):
    """
    RMSE of always predicting the training mean.

    Any useful model must beat this. It gives the reported RMSE a reference
    point, which is what section 5 asks about.
    """
    rmse = float(np.sqrt(np.mean((y_train - y_train.mean()) ** 2)))
    print("\nBaseline (always predict the training mean): RMSE %.2f" % rmse)
    return rmse


# --- Entry point -------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="TP1 - linear regression with k-fold CV")
    parser.add_argument("--data", default="insurance.csv")
    parser.add_argument("--folds", type=int, default=N_SPLITS)
    args = parser.parse_args()

    X, y, numeric_columns = prepare(args.data)

    X_train, X_test, y_train, y_test = split_data(X, y)
    print("=" * 68)
    print("SECTION 2 - LINEAR REGRESSION")
    print("=" * 68)
    print("2.1 Train/test split (random, %d%% test, seed %d)"
          % (TEST_SIZE * 100, RANDOM_STATE))
    print("  train: %d rows      test: %d rows (held out, not used below)"
          % (len(X_train), len(X_test)))

    baseline_rmse(y_train)
    cross_validate_rmse(X_train, y_train, numeric_columns,
                        n_splits=args.folds)
    show_coefficients(X_train, y_train, numeric_columns)

    print("\n" + "=" * 68)
    print("The test set was not evaluated. It is used once, in section 5.")
    print("=" * 68)


if __name__ == "__main__":
    main()
