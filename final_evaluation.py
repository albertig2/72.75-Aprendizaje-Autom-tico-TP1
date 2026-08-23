"""
TP1 - Regression and Introduction to Model Evaluation
Step 5: Model comparison and final evaluation on the test set.

This script answers the three questions of section 5:

  1. Which model obtained the lowest error?
  2. Which model would be deployed in a real application, and why?
  3. What RMSE would be promised on new data?

The test set is evaluated HERE AND ONLY HERE, once, with the single model
selected on the basis of the cross-validation results from sections 3 and 4.
The selection was made before this script ran; nothing below is allowed to
change it. Trying several models on the test set and reporting the best one
would turn the test set into a second validation set, and the number reported
would again be optimistic.
"""

import argparse
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from data_prep import prepare
from linear_regression import RANDOM_STATE, split_data
from polynomial_regression import build_poly_pipeline

# --- The selected model ------------------------------------------------------
# Chosen from the section 4 table: lowest cross-validated RMSE.
SELECTED_DEGREE = 2
SELECTED_LAMBDA = 100.0
SELECTED_VAL_RMSE = 4905.32   # cross-validated, from section 4
BASELINE_VAL_RMSE = 11701.05  # predict the training mean
LINEAR_VAL_RMSE = 6123.65     # section 2


def fit_selected_model(X_train, y_train, numeric_columns):
    """Refit the chosen model on the FULL training set (all 1069 rows).

    During cross-validation each model saw only ~855 rows. Refitting on the
    whole training set uses all the data available before the test set, which
    can only help: more data, same hyperparameters.
    """
    model = Lasso(alpha=SELECTED_LAMBDA, max_iter=100000, tol=1e-3,
                  random_state=RANDOM_STATE)
    pipeline = build_poly_pipeline(numeric_columns, model,
                                   degree=SELECTED_DEGREE)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        pipeline.fit(X_train, y_train)
    return pipeline


def bootstrap_rmse_interval(y_true, y_pred, n_boot=2000, alpha=0.05,
                            seed=RANDOM_STATE):
    """
    Percentile bootstrap interval for the test RMSE.

    The test RMSE is itself an estimate computed on 268 rows, so it carries
    sampling uncertainty. Quoting a single number without that uncertainty
    would overstate how precisely the performance is known.
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    errors = y_true - y_pred
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        stats.append(np.sqrt(np.mean(errors[idx] ** 2)))
    low, high = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(low), float(high)


def evaluate_on_test(pipeline, X_test, y_test):
    """The single, final evaluation."""
    y_pred = pipeline.predict(X_test)

    rmse = root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    low, high = bootstrap_rmse_interval(y_test, y_pred)

    print("\n" + "=" * 68)
    print("FINAL TEST EVALUATION - performed once, on %d unseen rows"
          % len(y_test))
    print("=" * 68)
    print("  Model : Lasso, polynomial degree %d, lambda %.1f"
          % (SELECTED_DEGREE, SELECTED_LAMBDA))
    print()
    print("  Test RMSE : %9.2f    95%% bootstrap interval [%.0f, %.0f]"
          % (rmse, low, high))
    print("  Test MAE  : %9.2f" % mae)
    print("  Test R2   : %9.4f" % r2)
    return rmse, mae, r2, (low, high), y_pred


def residual_analysis(X_test, y_test, y_pred):
    """
    Where the error actually lives.

    RMSE is a single number over a population that is not homogeneous. Showing
    the breakdown is what turns a number into an argument about deployment.
    """
    resid = y_test - y_pred
    df = pd.DataFrame({
        "smoker": np.where(X_test["smoker_yes"] == 1, "smoker", "non-smoker"),
        "y_true": y_test,
        "y_pred": y_pred,
        "residual": resid,
    })

    print("\n" + "-" * 68)
    print("Residual breakdown by subgroup")
    print("-" * 68)
    rows = []
    for name, group in df.groupby("smoker"):
        rows.append({
            "group": name,
            "n": len(group),
            "mean_actual": group["y_true"].mean(),
            "rmse": np.sqrt(np.mean(group["residual"] ** 2)),
            "mean_error": group["residual"].mean(),
        })
    summary = pd.DataFrame(rows)
    print(summary.round(2).to_string(index=False))

    over = int((resid < 0).sum())
    print("\n  The model over-predicts on %d of %d rows (%.0f%%) and "
          "under-predicts on the rest." % (over, len(resid),
                                           100 * over / len(resid)))
    print("  Largest under-prediction: %.0f" % resid.max())
    print("  Largest over-prediction : %.0f" % abs(resid.min()))
    return summary


def answer_questions(test_rmse, interval):
    """Section 5, questions 1 to 3."""
    low, high = interval

    print("\n" + "=" * 68)
    print("5. MODEL COMPARISON - answers")
    print("=" * 68)

    print("\nQ1. Which model obtained the lowest error?")
    print("-" * 68)
    print("  Lasso on degree-2 polynomial features with lambda = 100,")
    print("  cross-validated RMSE %.2f on the training set." % SELECTED_VAL_RMSE)
    print()
    print("  Ranking by cross-validated validation RMSE:")
    print("    Lasso  degree 2, lambda  100   ->  4905.32   <- best")
    print("    Lasso  degree 3, lambda  100   ->  4918.71")
    print("    OLS    degree 2                ->  4931.59")
    print("    OLS    degree 3                ->  5169.48")
    print("    OLS    degree 1 (linear)       ->  6123.65")
    print("    Baseline (training mean)       -> 11701.05")
    print()
    print("  Two observations matter more than the ranking itself:")
    print("   - Going from degree 1 to degree 2 cut the error by about 20%.")
    print("     The degree-2 expansion creates interaction terms such as")
    print("     bmi x smoker_yes, and that interaction is real: a high BMI")
    print("     raises costs far more for smokers than for non-smokers. An")
    print("     additive linear model cannot represent this at all.")
    print("   - Degree 3 made validation error WORSE (5169 vs 4932) while")
    print("     training error kept falling (4514 vs 4749). The train/val")
    print("     gap grew from 183 to 655. That is overfitting: 164 features")
    print("     fitted on ~855 rows start memorising noise.")

    print("\nQ2. Which model would you deploy in a real application?")
    print("-" * 68)
    print("  Lasso, degree 2, lambda = 100.")
    print()
    print("  Honest caveat first: its advantage over plain OLS on degree-2")
    print("  features is %.2f RMSE (4905.32 vs 4931.59), while the fold-to-"
          % (4931.59 - SELECTED_VAL_RMSE))
    print("  fold standard deviation is around 200. That difference is NOT")
    print("  statistically meaningful. The two models are equivalent in")
    print("  accuracy, so accuracy alone does not decide between them.")
    print()
    print("  It is chosen on the other three grounds:")
    print("   - Stability: its train/val gap is 108 versus 183 for OLS. It")
    print("     depends less on which particular rows it was trained on,")
    print("     which is what matters when new data arrives.")
    print("   - Simplicity: L1 sets 24 of the 44 coefficients to exactly")
    print("     zero, so the deployed model uses about half the features.")
    print("     Fewer active terms means less to compute, less to maintain,")
    print("     and less that can silently break.")
    print("   - Robustness of the choice: lambda = 100 is not a knife edge.")
    print("     Values from 1 to 100 all give validation RMSE near 4900, so")
    print("     performance does not hinge on tuning lambda exactly right.")

    print("\nQ3. What RMSE would you promise on new data?")
    print("-" * 68)
    print("  Approximately %.0f, and honestly stated as a range of roughly"
          % test_rmse)
    print("  %.0f to %.0f rather than a single figure." % (low, high))
    print()
    print("  Why NOT quote the cross-validation number (%.2f):"
          % SELECTED_VAL_RMSE)
    print("   The validation RMSE was used to CHOOSE the degree and lambda.")
    print("   Sixteen model configurations were compared and the best one was")
    print("   kept. Selecting the minimum of sixteen noisy estimates biases")
    print("   that minimum downward: part of why it won is genuine quality,")
    print("   and part is that its folds happened to be favourable. So the")
    print("   validation RMSE is no longer an unbiased estimate of future")
    print("   performance - it has been optimised against.")
    print()
    print("   The test set was never involved in any of those decisions, so")
    print("   the test RMSE is the only honest estimate available.")
    print()
    print("  Two conditions attached to the promise:")
    print("   - It assumes new data resembles this data: US insurance data,")
    print("     ages 18-64, same cost structure. It does not transfer to a")
    print("     different country, a different year, or a different insurer.")
    print("   - The error is not evenly distributed. See the subgroup")
    print("     breakdown above: the model is far more accurate for")
    print("     non-smokers than for smokers. A single average RMSE hides")
    print("     that, and any real deployment should quote both.")


def main():
    parser = argparse.ArgumentParser(
        description="TP1 - final evaluation on the test set")
    parser.add_argument("--data", default="insurance.csv")
    args = parser.parse_args()

    X, y, numeric_columns = prepare(args.data)
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("=" * 68)
    print("SECTION 5 - MODEL COMPARISON AND FINAL EVALUATION")
    print("=" * 68)
    print("Model selected from the section 4 cross-validation table:")
    print("  Lasso, degree %d, lambda %.1f (CV RMSE %.2f)"
          % (SELECTED_DEGREE, SELECTED_LAMBDA, SELECTED_VAL_RMSE))
    print("Refitting on all %d training rows, then evaluating once on the"
          % len(X_train))
    print("%d test rows that have been held out since section 2.1."
          % len(X_test))

    pipeline = fit_selected_model(X_train, y_train, numeric_columns)
    test_rmse, mae, r2, interval, y_pred = evaluate_on_test(
        pipeline, X_test, y_test)

    print("\n  For comparison, cross-validated on the training set: %.2f"
          % SELECTED_VAL_RMSE)
    print("  Difference (test - validation): %+.2f" % (test_rmse - SELECTED_VAL_RMSE))

    residual_analysis(X_test, y_test, y_pred)
    answer_questions(test_rmse, interval)

    print("\n" + "=" * 68)
    print("The test set has now been used. It must not be used again to")
    print("compare further models - doing so would invalidate this estimate.")
    print("=" * 68)


if __name__ == "__main__":
    main()
