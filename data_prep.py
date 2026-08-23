"""
TP1 - Regression and Introduction to Model Evaluation
Step 1: Data cleaning and preparation.

This module loads the insurance dataset, reports the diagnostics needed to
justify the decisions in section 1 of the report, and returns an encoded
feature matrix ready for modelling.

Scaling is deliberately NOT done here. The scaler must be fitted inside each
cross-validation fold (on the training portion only), otherwise information
from the validation fold leaks into the preprocessing. See section 2.
"""

import argparse

import numpy as np
import pandas as pd

# --- Column groups -----------------------------------------------------------

TARGET = "charges"

NUMERIC_FEATURES = ["age", "bmi", "children"]
BINARY_FEATURES = ["sex", "smoker"]
NOMINAL_FEATURES = ["region"]

# Explicit level -> 1 mapping so the encoding is documented, not implicit.
BINARY_POSITIVE_LEVEL = {"sex": "male", "smoker": "yes"}


# --- Loading -----------------------------------------------------------------

def load_data(path="insurance.csv"):
    """Read the raw CSV exactly as delivered, with no modifications."""
    return pd.read_csv(path)


# --- Section 1 diagnostics ---------------------------------------------------

def report_missing(df):
    """1.2 - Missing values."""
    missing = df.isna().sum()
    total = int(missing.sum())
    print("1.2 Missing values")
    if total == 0:
        print("  No missing values in any of the %d columns." % df.shape[1])
        print("  No imputation strategy is required.")
    else:
        print(missing[missing > 0].to_string())
    return total


def report_duplicates(df):
    """1.2 - Exact duplicate rows."""
    dup_mask = df.duplicated(keep=False)
    n_extra = int(df.duplicated().sum())
    print("\n1.2b Duplicate rows")
    print("  Exact duplicate rows to drop: %d" % n_extra)
    if n_extra:
        print(df[dup_mask].to_string())
        print("  Dropped: an identical row appearing in both a training and a")
        print("  validation fold would leak information across the split.")
    return n_extra


def iqr_bounds(series, k=1.5):
    """Tukey's fences. Returns (lower, upper)."""
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def report_outliers(df, k=1.5):
    """1.3 - Outliers in the numeric variables, using the 1.5*IQR criterion."""
    print("\n1.3 Outliers (Tukey %.1f*IQR criterion)" % k)
    rows = []
    for col in NUMERIC_FEATURES + [TARGET]:
        low, high = iqr_bounds(df[col], k)
        mask = (df[col] < low) | (df[col] > high)
        rows.append({
            "variable": col,
            "lower": round(low, 2),
            "upper": round(high, 2),
            "n_outliers": int(mask.sum()),
            "pct": round(100 * mask.mean(), 1),
        })
    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False))

    # The key justification: the high-charge "outliers" are almost all smokers,
    # so they are a real subpopulation, not measurement error.
    _, high = iqr_bounds(df[TARGET], k)
    extreme = df[df[TARGET] > high]
    if len(extreme):
        share_extreme = 100 * (extreme["smoker"] == "yes").mean()
        share_overall = 100 * (df["smoker"] == "yes").mean()
        print("\n  Of the %d high-%s rows, %.1f%% are smokers," %
              (len(extreme), TARGET, share_extreme))
        print("  versus %.1f%% in the dataset overall." % share_overall)
        print("  Decision: KEEP them. They are a genuine subpopulation the")
        print("  model must learn, not data-entry errors. Removing them would")
        print("  delete real signal and make the reported RMSE optimistic.")
    return summary


def report_categoricals(df):
    """1.1 - Categorical variables and their levels."""
    print("\n1.1 Categorical variables")
    for col in BINARY_FEATURES + NOMINAL_FEATURES:
        counts = df[col].value_counts().to_dict()
        kind = "binary" if col in BINARY_FEATURES else "nominal"
        print("  %-8s (%s, %d levels): %s" % (col, kind, len(counts), counts))
    print("  'children' is left as a numeric count (0-5), not one-hot encoded.")


def report_target(df):
    """Context for the error metric discussion in section 5."""
    y = df[TARGET]
    print("\nTarget variable (%s)" % TARGET)
    print("  mean=%.2f  std=%.2f  min=%.2f  max=%.2f  skew=%.3f" %
          (y.mean(), y.std(), y.min(), y.max(), y.skew()))
    print("  Right-skewed. Kept in original units so RMSE is interpretable")
    print("  in pesos/dollars, as the assignment requires.")


def run_diagnostics(df):
    """Print every check behind the section 1 decisions."""
    print("=" * 68)
    print("SECTION 1 DIAGNOSTICS - raw dataset: %d rows x %d columns"
          % df.shape)
    print("=" * 68)
    report_categoricals(df)
    report_missing(df)
    report_duplicates(df)
    report_outliers(df)
    report_target(df)
    print("=" * 68)


# --- Cleaning and encoding ---------------------------------------------------

def clean_data(df):
    """Apply the section 1 decisions: drop exact duplicates, keep outliers."""
    return df.drop_duplicates().reset_index(drop=True)


def encode_features(df):
    """
    Turn the raw columns into a numeric design matrix.

    - binary columns  -> single 0/1 column (one-hot would be redundant)
    - 'region'        -> one-hot with the first level dropped, to avoid
                         perfect collinearity with the intercept
    - numeric columns -> passed through unchanged (scaled later, inside CV)

    Returns (X, y, numeric_columns).
    """
    out = df.copy()

    for col in BINARY_FEATURES:
        positive = BINARY_POSITIVE_LEVEL[col]
        out[col] = (out[col] == positive).astype(int)
        out = out.rename(columns={col: f"{col}_{positive}"})

    out = pd.get_dummies(
        out, columns=NOMINAL_FEATURES, drop_first=True, dtype=int
    )

    y = out[TARGET].to_numpy(dtype=float)
    X = out.drop(columns=[TARGET])

    # Columns that need standardisation later. The 0/1 dummies do not.
    numeric_columns = [c for c in NUMERIC_FEATURES if c in X.columns]
    return X, y, numeric_columns


def prepare(path="insurance.csv", verbose=False):
    """Full pipeline: load -> (diagnostics) -> clean -> encode."""
    df = load_data(path)
    if verbose:
        run_diagnostics(df)
    df = clean_data(df)
    X, y, numeric_columns = encode_features(df)
    return X, y, numeric_columns


# --- Entry point -------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="TP1 - data preparation")
    parser.add_argument("--data", default="insurance.csv",
                        help="path to insurance.csv")
    args = parser.parse_args()

    X, y, numeric_columns = prepare(args.data, verbose=True)

    print("\nEncoded design matrix: %d rows x %d features" % X.shape)
    print("  features: %s" % ", ".join(X.columns))
    print("  to be standardised inside each CV fold: %s"
          % ", ".join(numeric_columns))
    print("\nFirst 5 rows:")
    print(X.head().to_string(index=False))
    print("\nTarget: %d values, mean %.2f" % (len(y), float(np.mean(y))))


if __name__ == "__main__":
    main()
