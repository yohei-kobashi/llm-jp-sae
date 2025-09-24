"""
Usage
-----
1) Put your Parquet files under:  modality_data/
   Required columns in each file:
       - "Modal_Verb": target class (string)
       - "sae_tokens": list (or ndarray) of tokens (string)
       - "sae_latent_indices": list (or ndarray) of list/ndarray[int] per token
       - "sae_latent_acts":    list (or ndarray) of list/ndarray[float] per token

2) Optional configuration (see "Configuration" section below):
       - PARQUET_GLOB: glob pattern of input Parquet files
       - RESULT_CSV:   output CSV for results
       - TEST_SIZE:    train/test split ratio
       - USE_MODAL_TOKEN_ONLY:
           * False (default): pool activations across ALL tokens in a sentence
           * True:            use ONLY the token(s) exactly equal to the modal verb
       - LOGREG_C, MAX_ITER: logistic regression hyperparameters

3) Run:
       python predict_modality.py

4) The script will:
       - Iterate all Parquet files that match PARQUET_GLOB
       - Build a CSR sparse feature matrix X per file
       - Label-encode "Modal_Verb" into y
       - Train multinomial logistic regression with L1 regularization (saga)
       - Report train/test accuracy per file
       - Save a CSV summary to RESULT_CSV
"""

import glob
import os
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, vstack
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# ======================
# Configuration
# ======================
PARQUET_GLOB = "modality_data/*.parquet"
RESULT_CSV = "results_modality_logreg.csv"
TEST_SIZE = 0.2
RANDOM_STATE = 42
USE_MODAL_TOKEN_ONLY = False   # True: only use tokens equal to the current "Modal_Verb"
LOGREG_C = 1.0
MAX_ITER = 2000
CLASS_WEIGHT = None            # e.g., "balanced" if you have strong class imbalance


# ======================
# Utilities (robust to ndarray/list/None)
# ======================
def is_empty_seq(x):
    """Return True if x is None or has length 0 (works for list/ndarray)."""
    if x is None:
        return True
    try:
        return len(x) == 0
    except TypeError:
        return False


def as_list1d(x):
    """Convert 1D ndarray -> list; keep list as-is; None -> []."""
    if x is None:
        return []
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x


def as_list2d(x):
    """
    Convert nested ndarray/list -> list[list]; None -> [].
    Ensures the outer container is a list where each element is a list (1D).
    """
    if x is None:
        return []
    if isinstance(x, np.ndarray):
        return [as_list1d(row) for row in x]
    # Assume x is already an iterable of rows
    return [as_list1d(row) for row in x]


# ======================
# Core helpers
# ======================
def max_idx_from_nested(nested_idx_col):
    """
    Find the maximum index across a column that stores nested index lists
    (shape: List[List[int]] or ndarray-of-lists). Returns -1 if no indices exist.
    """
    mx = -1
    for sent_idxs in nested_idx_col:
        sent_idxs = as_list2d(sent_idxs)
        if is_empty_seq(sent_idxs):
            continue
        for idxs in sent_idxs:
            idxs = as_list1d(idxs)
            if is_empty_seq(idxs):
                continue
            local_max = max(idxs)
            if local_max > mx:
                mx = local_max
    return mx


def sentence_pool_sum(indices_per_token, acts_per_token, D):
    """
    Sum pool across ALL tokens in a sentence and return a 1xD CSR row.
    Robust to None/ndarray/list and to per-token length mismatches.
    """
    indices_per_token = as_list2d(indices_per_token)
    acts_per_token    = as_list2d(acts_per_token)

    if is_empty_seq(indices_per_token) or is_empty_seq(acts_per_token):
        return csr_matrix((1, D), dtype=np.float32)

    rows_idx, rows_val = [], []
    for idxs, vals in zip(indices_per_token, acts_per_token):
        idxs = as_list1d(idxs)
        vals = as_list1d(vals)
        if is_empty_seq(idxs) or is_empty_seq(vals):
            continue
        if len(idxs) != len(vals):
            m = min(len(idxs), len(vals))
            idxs, vals = idxs[:m], vals[:m]
        rows_idx.extend(idxs)
        rows_val.extend(vals)

    if len(rows_idx) == 0:
        return csr_matrix((1, D), dtype=np.float32)

    cnt = np.bincount(
        np.asarray(rows_idx, dtype=np.int64),
        weights=np.asarray(rows_val, dtype=np.float32),
        minlength=D,
    )
    nz = np.nonzero(cnt)[0]
    return csr_matrix((cnt[nz], (np.zeros_like(nz), nz)), shape=(1, D), dtype=np.float32)


def sentence_pool_sum_modal_only(tokens, indices_per_token, acts_per_token, target_modal, D):
    """
    Sum pool restricted to tokens whose surface form exactly equals `target_modal`.
    Returns a 1xD CSR row. Robust to None/ndarray/list and to per-token length mismatches.
    """
    tokens            = as_list1d(tokens)
    indices_per_token = as_list2d(indices_per_token)
    acts_per_token    = as_list2d(acts_per_token)

    if is_empty_seq(tokens) or is_empty_seq(indices_per_token) or is_empty_seq(acts_per_token):
        return csr_matrix((1, D), dtype=np.float32)

    rows_idx, rows_val = [], []
    for tok, idxs, vals in zip(tokens, indices_per_token, acts_per_token):
        idxs = as_list1d(idxs)
        vals = as_list1d(vals)
        if (tok == target_modal) and (not is_empty_seq(idxs)) and (not is_empty_seq(vals)):
            if len(idxs) != len(vals):
                m = min(len(idxs), len(vals))
                idxs, vals = idxs[:m], vals[:m]
            rows_idx.extend(idxs)
            rows_val.extend(vals)

    if len(rows_idx) == 0:
        return csr_matrix((1, D), dtype=np.float32)

    cnt = np.bincount(
        np.asarray(rows_idx, dtype=np.int64),
        weights=np.asarray(rows_val, dtype=np.float32),
        minlength=D,
    )
    nz = np.nonzero(cnt)[0]
    return csr_matrix((cnt[nz], (np.zeros_like(nz), nz)), shape=(1, D), dtype=np.float32)

def build_X_y(df):
    """Build the CSR feature matrix X and label vector y from a DataFrame."""
    print("  [build_X_y] Start")
    # Ensure label is string
    modal_str = df["Modal_Verb"].astype(str)
    le = LabelEncoder()
    y = le.fit_transform(modal_str)
    print(f"  [build_X_y] Modal_Verb classes: {list(le.classes_)}")

    # Estimate SAE dictionary size
    D = max_idx_from_nested(df["sae_latent_indices"]) + 1
    print(f"  [build_X_y] Estimated dictionary size D={D}")
    if D <= 0:
        raise ValueError("Dictionary size D could not be estimated (no non-empty activations).")

    # Build CSR rows
    rows = []
    print(f"  [build_X_y] Building rows, total samples={len(df)} ...")
    if USE_MODAL_TOKEN_ONLY:
        for i, (tokens, idxs, vals, modal) in enumerate(
            zip(df["sae_tokens"], df["sae_latent_indices"], df["sae_latent_acts"], modal_str)
        ):
            if i % 500 == 0:
                print(f"    processed {i} rows...")
            row = sentence_pool_sum_modal_only(tokens, idxs, vals, target_modal=str(modal), D=D)
            rows.append(row)
    else:
        for i, (idxs, vals) in enumerate(zip(df["sae_latent_indices"], df["sae_latent_acts"])):
            if i % 500 == 0:
                print(f"    processed {i} rows...")
            row = sentence_pool_sum(indices_per_token=idxs, acts_per_token=vals, D=D)
            rows.append(row)

    X = vstack(rows, format="csr")
    print(f"  [build_X_y] Finished building X with shape {X.shape}, nnz={X.nnz}")
    return X, y, D, le


def run_one_file(path):
    """Process one Parquet file with debug logs."""
    print(f"\n[run_one_file] Processing {path}")
    df = pd.read_parquet(path)
    print(f"  Rows loaded: {len(df)}")

    required = ["Modal_Verb", "sae_tokens", "sae_latent_indices", "sae_latent_acts"]
    for col in required:
        if col not in df.columns:
            raise KeyError(f"{path}: required column {col} is missing.")

    X, y, D, le = build_X_y(df)

    unique_classes = np.unique(y)
    print(f"  [run_one_file] #classes={len(unique_classes)}")

    if len(unique_classes) < 2:
        return {
            "file": os.path.basename(path),
            "n_samples": X.shape[0],
            "n_features": D,
            "n_classes": len(unique_classes),
            "train_acc": np.nan,
            "test_acc": np.nan,
            "note": "Skipped (only one class present)",
        }

    print("  [run_one_file] Splitting train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    print(f"  [run_one_file] Train shape={X_train.shape}, Test shape={X_test.shape}")

    print("  [run_one_file] Training logistic regression...")
    clf = LogisticRegression(
        multi_class="multinomial",
        solver="saga",
        penalty="l1",
        C=LOGREG_C,
        max_iter=MAX_ITER,
        n_jobs=-1,
        class_weight=CLASS_WEIGHT,
        verbose=1,   # scikit-learn’s own progress log
    )
    clf.fit(X_train, y_train)
    print("  [run_one_file] Training complete")

    train_acc = clf.score(X_train, y_train)
    test_acc = clf.score(X_test, y_test)
    print(f"  [run_one_file] Train acc={train_acc:.3f}, Test acc={test_acc:.3f}")

    return {
        "file": os.path.basename(path),
        "n_samples": X.shape[0],
        "n_features": D,
        "n_classes": len(le.classes_),
        "train_acc": float(train_acc),
        "test_acc": float(test_acc),
        "note": "",
    }


def main():
    """Run logistic regression over all Parquet files and save a CSV summary."""
    paths = sorted(glob.glob(PARQUET_GLOB))
    if not paths:
        print(f"No Parquet files found for pattern: {PARQUET_GLOB}")
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
    print(f"Results saved to: {RESULT_CSV}")


if __name__ == "__main__":
    main()
