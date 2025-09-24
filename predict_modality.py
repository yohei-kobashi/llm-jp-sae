"""
Usage:
------
1. Place all your Parquet files inside the directory `modality_data/`.
   Each file must contain at least the following columns:
       - "Modal_Verb": the target class (string)
       - "sae_tokens": list of tokens (string)
       - "sae_latent_indices": list of list of ints (indices of active features)
       - "sae_latent_acts": list of list of floats (activation values)
2. Adjust the global variables if needed:
       - PARQUET_GLOB: glob pattern for input files
       - RESULT_CSV: output CSV file name
       - TEST_SIZE: train/test split ratio
       - USE_MODAL_TOKEN_ONLY: True = only use the modal verb token activations,
                               False = pool all tokens in the sentence
3. Run the script:
       python this_script.py
4. The script will:
       - Iterate over all Parquet files matching the glob
       - Build feature matrices (CSR sparse) and label arrays
       - Train multinomial logistic regression with L1 regularization
       - Evaluate accuracy on train/test split
       - Save results into a CSV file (RESULT_CSV)
"""

import glob
import os
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, vstack
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# ==== Configuration ====
PARQUET_GLOB = "modality_data/*.parquet"
RESULT_CSV = "results_modality_logreg.csv"
TEST_SIZE = 0.2
RANDOM_STATE = 42
USE_MODAL_TOKEN_ONLY = False        # True: use only modal verb token activations
LOGREG_C = 1.0
MAX_ITER = 2000


def max_idx_from_nested(nested_idx_col):
    """Return maximum index across a nested list column of indices."""
    mx = -1
    for sent_idxs in nested_idx_col:
        if not sent_idxs:
            continue
        for idxs in sent_idxs:
            if idxs:
                local_max = max(idxs)
                if local_max > mx:
                    mx = local_max
    return mx


def sentence_pool_sum(indices_per_token, acts_per_token, D):
    """Sum pooling over all tokens in a sentence. Returns a 1xD CSR row."""
    if not indices_per_token or not acts_per_token:
        return csr_matrix((1, D), dtype=np.float32)
    rows_idx, rows_val = [], []
    for idxs, vals in zip(indices_per_token, acts_per_token):
        if not idxs:
            continue
        rows_idx.extend(idxs)
        rows_val.extend(vals)
    if not rows_idx:
        return csr_matrix((1, D), dtype=np.float32)
    cnt = np.bincount(
        np.asarray(rows_idx, dtype=np.int64),
        weights=np.asarray(rows_val, dtype=np.float32),
        minlength=D,
    )
    nz = np.nonzero(cnt)[0]
    return csr_matrix((cnt[nz], (np.zeros_like(nz), nz)), shape=(1, D), dtype=np.float32)


def sentence_pool_sum_modal_only(tokens, indices_per_token, acts_per_token, target_modal, D):
    """Sum pooling restricted to tokens matching the modal verb only."""
    if not tokens or not indices_per_token or not acts_per_token:
        return csr_matrix((1, D), dtype=np.float32)
    rows_idx, rows_val = [], []
    for tok, idxs, vals in zip(tokens, indices_per_token, acts_per_token):
        if tok == target_modal and idxs:
            rows_idx.extend(idxs)
            rows_val.extend(vals)
    if not rows_idx:
        return csr_matrix((1, D), dtype=np.float32)
    cnt = np.bincount(
        np.asarray(rows_idx, dtype=np.int64),
        weights=np.asarray(rows_val, dtype=np.float32),
        minlength=D,
    )
    nz = np.nonzero(cnt)[0]
    return csr_matrix((cnt[nz], (np.zeros_like(nz), nz)), shape=(1, D), dtype=np.float32)


def build_X_y(df):
    """Build feature matrix X and label vector y from a DataFrame."""
    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(df["Modal_Verb"].astype(str))

    # Estimate dictionary size
    D = max_idx_from_nested(df["sae_latent_indices"]) + 1
    if D <= 0:
        raise ValueError("Dictionary size D could not be estimated (no activations found).")

    # Build sparse rows
    rows = []
    if USE_MODAL_TOKEN_ONLY:
        for tokens, idxs, vals, modal in zip(
            df["sae_tokens"], df["sae_latent_indices"], df["sae_latent_acts"], df["Modal_Verb"]
        ):
            row = sentence_pool_sum_modal_only(tokens, idxs, vals, target_modal=str(modal), D=D)
            rows.append(row)
    else:
        for idxs, vals in zip(df["sae_latent_indices"], df["sae_latent_acts"]):
            row = sentence_pool_sum(indices_per_token=idxs, acts_per_token=vals, D=D)
            rows.append(row)

    X = vstack(rows, format="csr")
    return X, y, D, le


def run_one_file(path):
    """Process one Parquet file: train/test split, fit logistic regression, return results dict."""
    df = pd.read_parquet(path)
    required = ["Modal_Verb", "sae_tokens", "sae_latent_indices", "sae_latent_acts"]
    for col in required:
        if col not in df.columns:
            raise KeyError(f"{path}: required column {col} is missing.")

    X, y, D, le = build_X_y(df)

    # Skip if only one class
    if len(np.unique(y)) < 2:
        return {
            "file": os.path.basename(path),
            "n_samples": X.shape[0],
            "n_features": D,
            "n_classes": len(np.unique(y)),
            "train_acc": np.nan,
            "test_acc": np.nan,
            "note": "Skipped (only one class present)",
        }

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    clf = LogisticRegression(
        multi_class="multinomial",
        solver="saga",
        penalty="l1",
        C=LOGREG_C,
        max_iter=MAX_ITER,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)

    return {
        "file": os.path.basename(path),
        "n_samples": X.shape[0],
        "n_features": D,
        "n_classes": len(le.classes_),
        "train_acc": train_acc,
        "test_acc": test_acc,
        "note": "",
    }


def main():
    """Run logistic regression on all parquet files and save results to CSV."""
    paths = sorted(glob.glob(PARQUET_GLOB))
    if not paths:
        print(f"No Parquet files found for pattern {PARQUET_GLOB}")
        return

    results = []
    for p in paths:
        try:
            res = run_one_file(p)
        except Exception as e:
            res = {
                "file": os.path.basename(p),
                "n_samples": np.nan,
                "n_features": np.nan,
                "n_classes": np.nan,
                "train_acc": np.nan,
                "test_acc": np.nan,
                "note": f"ERROR: {type(e).__name__}: {e}",
            }
        print(res)
        results.append(res)

    out_df = pd.DataFrame(results)
    out_df.to_csv(RESULT_CSV, index=False, encoding="utf-8")
    print(f"Results saved to {RESULT_CSV}")


if __name__ == "__main__":
    main()
