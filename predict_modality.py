"""
predict_modality_parallel.py

What this script does
---------------------
- Loads all Parquet files under `modality_data/`
- For each file and each target label option:
    - Builds sparse features from SAE activations (token-pooled)
    - Filters labels: removes "unknown" (where applicable) and drops rare classes (n < 5)
    - Trains multinomial logistic regression with L1 (saga) for a small HP grid:
          C fixed to 1.0
          (max_iter, tol) in {(200, 1e-3), (500, 1e-4)}
          class_weight in {None, "balanced"}
    - Reports train/test accuracy and the number of nonzero-coefficient features
- Runs tasks in parallel using processes
- Writes a combined CSV summary

Run
---
    python predict_modality_parallel.py

Expected columns in Parquet
---------------------------
- "Modal_Verb" (string)
- "Palmer_Expected" (string; can include "unknown")
- "Quirk_Expected"  (string; can include "unknown")
- "sae_tokens"          (list/ndarray[str])
- "sae_latent_indices"  (list/ndarray[list/ndarray[int]])
- "sae_latent_acts"     (list/ndarray[list/ndarray[float]])
"""

import os
# Avoid thread oversubscription inside each process (MKL/OpenBLAS/NumExpr/OpenMP)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import glob
import time
import concurrent.futures as cf
import warnings
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, vstack
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning

# ======================
# Configuration
# ======================
PARQUET_GLOB = "modality_data/*.parquet"
RESULT_CSV = "results_modality_logreg_all_targets_min_tune_parallel.csv"

# Train/test split
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Feature pooling
USE_MODAL_TOKEN_ONLY = False   # True: only tokens equal to surface Modal_Verb are pooled

# Logistic regression core settings
SOLVER = "saga"
PENALTY = "l1"
N_JOBS = 1                     # Important: 1 per process; parallelism is at process level
MULTI_CLASS = "auto"           # Quiet deprecation; >=1.7 will default to multinomial

# Minimal hyperparameter schedule
Cs = [0.25, 0.5, 1]
HP_SCHEDULE = [
    {"max_iter": 1000, "tol": 1e-4},
    {"max_iter": 2000, "tol": 1e-4},
    {"max_iter": 3000, "tol": 1e-4},
    {"max_iter": 4000, "tol": 1e-4},
    {"max_iter": 5000, "tol": 1e-4},
]
CLASS_WEIGHTS = [None]   # try with/without balancing

# All target options to run
TARGET_OPTIONS = [
    "modal_verb",
    "palmer_expected",
    "quirk_expected",
    "palmer_quirk_pair",
]
UNKNOWN_TOKEN = "unknown"      # case-insensitive

# Rare-class pruning threshold: drop classes with < MIN_CLASS_SAMPLES
MIN_CLASS_SAMPLES = 5

# ======================
# Utilities (robust to ndarray/list/None)
# ======================
def is_empty_seq(x):
    if x is None:
        return True
    try:
        return len(x) == 0
    except TypeError:
        return False

def as_list1d(x):
    if x is None:
        return []
    if isinstance(x, np.ndarray):
        return x.tolist()
    return x

def as_list2d(x):
    if x is None:
        return []
    if isinstance(x, np.ndarray):
        return [as_list1d(row) for row in x]
    return [as_list1d(row) for row in x]

# ======================
# Core helpers
# ======================
def max_idx_from_nested(nested_idx_col):
    """Max index across nested index lists. Returns -1 if none."""
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
    """Sum-pool activations across ALL tokens -> 1xD CSR row."""
    indices_per_token = as_list2d(indices_per_token)
    acts_per_token    = as_list2d(acts_per_token)
    if is_empty_seq(indices_per_token) or is_empty_seq(acts_per_token):
        return csr_matrix((1, D), dtype=np.float32)

    rows_idx, rows_val = [], []
    for idxs, vals in zip(indices_per_token, acts_per_token):
        idxs = as_list1d(idxs); vals = as_list1d(vals)
        if is_empty_seq(idxs) or is_empty_seq(vals):
            continue
        if len(idxs) != len(vals):
            m = min(len(idxs), len(vals))
            idxs, vals = idxs[:m], vals[:m]
        rows_idx.extend(idxs); rows_val.extend(vals)

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
    """Sum-pool restricted to tokens whose surface form equals target_modal."""
    tokens            = as_list1d(tokens)
    indices_per_token = as_list2d(indices_per_token)
    acts_per_token    = as_list2d(acts_per_token)
    if is_empty_seq(tokens) or is_empty_seq(indices_per_token) or is_empty_seq(acts_per_token):
        return csr_matrix((1, D), dtype=np.float32)

    rows_idx, rows_val = [], []
    for tok, idxs, vals in zip(tokens, indices_per_token, acts_per_token):
        idxs = as_list1d(idxs); vals = as_list1d(vals)
        if (tok == target_modal) and (not is_empty_seq(idxs)) and (not is_empty_seq(vals)):
            if len(idxs) != len(vals):
                m = min(len(idxs), len(vals))
                idxs, vals = idxs[:m], vals[:m]
            rows_idx.extend(idxs); rows_val.extend(vals)

    if len(rows_idx) == 0:
        return csr_matrix((1, D), dtype=np.float32)

    cnt = np.bincount(
        np.asarray(rows_idx, dtype=np.int64),
        weights=np.asarray(rows_val, dtype=np.float32),
        minlength=D,
    )
    nz = np.nonzero(cnt)[0]
    return csr_matrix((cnt[nz], (np.zeros_like(nz), nz)), shape=(1, D), dtype=np.float32)

def select_and_encode_labels(df, target_option):
    """
    Build a filtered DataFrame and a string label series based on `target_option`.
    Steps:
      - filter out 'unknown' where applicable
      - drop rare classes with counts < MIN_CLASS_SAMPLES
    Returns: filtered_df, label_str_series, dropped_count, option_used
    """
    option = target_option.lower().strip()

    if option == "modal_verb":
        label = df["Modal_Verb"].astype(str)
        filt = pd.Series([True] * len(df), index=df.index)

    elif option == "palmer_expected":
        pal = df["Palmer_Expected"].astype(str)
        keep = pal.str.lower().str.strip() != UNKNOWN_TOKEN
        filt = keep.fillna(False)
        label = pal[filt]

    elif option == "quirk_expected":
        quirk = df["Quirk_Expected"].astype(str)
        keep = quirk.str.lower().str.strip() != UNKNOWN_TOKEN
        filt = keep.fillna(False)
        label = quirk[filt]

    elif option == "palmer_quirk_pair":
        pal = df["Palmer_Expected"].astype(str)
        quirk = df["Quirk_Expected"].astype(str)
        pal_ok = pal.str.lower().str.strip() != UNKNOWN_TOKEN
        quirk_ok = quirk.str.lower().str.strip() != UNKNOWN_TOKEN
        keep = pal_ok & quirk_ok
        filt = keep.fillna(False)
        label = (pal[filt].str.strip() + "|" + quirk[filt].str.strip())

    else:
        raise ValueError(f"Unknown TARGET_OPTION: {target_option}")

    # Filter "unknown"
    filtered_df = df[filt].reset_index(drop=True)
    label = label.reset_index(drop=True)
    dropped = int((~filt).sum())

    # Drop rare classes
    counts = label.value_counts()
    valid = counts[counts >= MIN_CLASS_SAMPLES].index
    keep_final = label.isin(valid)
    dropped_small = int((~keep_final).sum())

    filtered_df = filtered_df[keep_final].reset_index(drop=True)
    label = label[keep_final].reset_index(drop=True)
    dropped += dropped_small

    return filtered_df, label, dropped, option

def build_X_y(df, target_option):
    """
    Returns: X (csr), y (ndarray), D (int), label_encoder, n_used, n_dropped, classes(list[str])
    """
    df2, label_str, n_dropped, option_used = select_and_encode_labels(df, target_option)
    le = LabelEncoder()
    y = le.fit_transform(label_str.astype(str))

    D = max_idx_from_nested(df2["sae_latent_indices"]) + 1
    if D <= 0:
        raise ValueError("Dictionary size D could not be estimated (no non-empty activations).")

    rows = []
    if USE_MODAL_TOKEN_ONLY:
        for tokens, idxs, vals, modal_value in zip(
            df2["sae_tokens"], df2["sae_latent_indices"], df2["sae_latent_acts"], df2["Modal_Verb"].astype(str)
        ):
            row = sentence_pool_sum_modal_only(tokens, idxs, vals, target_modal=str(modal_value), D=D)
            rows.append(row)
    else:
        for idxs, vals in zip(df2["sae_latent_indices"], df2["sae_latent_acts"]):
            row = sentence_pool_sum(indices_per_token=idxs, acts_per_token=vals, D=D)
            rows.append(row)

    X = vstack(rows, format="csr")
    return X, y, D, le, len(df2), n_dropped, list(le.classes_)

# ---------- Model helpers ----------
def fit_logreg(X, y, c, tol, max_iter, class_weight):
    """Fit multinomial logistic regression with L1; suppress ConvergenceWarning."""
    clf = LogisticRegression(
        solver=SOLVER,
        penalty=PENALTY,
        C=c,
        tol=tol,
        max_iter=max_iter,
        n_jobs=N_JOBS,
        class_weight=class_weight,
        multi_class=MULTI_CLASS,
        verbose=0,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        clf.fit(X, y)
    return clf

def count_nnz_features(clf):
    """
    Count features with any nonzero coefficient across classes.
    coef_.shape == (n_classes, n_features)
    """
    w = clf.coef_
    nnz_mask = (np.abs(w) > 0).any(axis=0)
    return int(nnz_mask.sum())

# ---------- Worker ----------
def run_one_task(task):
    """
    One (file, target, hp, class_weight) combination -> result dict.
    """
    path, target_option, c, hp, class_weight = task
    file_name = os.path.basename(path)
    try:
        df = pd.read_parquet(path)

        required = [
            "Modal_Verb",
            "Palmer_Expected",
            "Quirk_Expected",
            "sae_tokens",
            "sae_latent_indices",
            "sae_latent_acts",
        ]
        for col in required:
            if col not in df.columns:
                raise KeyError(f"{path}: required column {col} is missing.")

        X, y, D, le, n_used, n_dropped, classes = build_X_y(df, target_option)
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            return {
                "file": file_name,
                "target": target_option,
                "C": c,
                "max_iter": hp["max_iter"],
                "tol": hp["tol"],
                "class_weight": class_weight if class_weight is not None else "none",
                "n_samples_used": n_used,
                "n_rows_dropped": n_dropped,
                "n_features": D,
                "n_classes": len(unique_classes),
                "train_acc": np.nan,
                "test_acc": np.nan,
                "nnz_features": np.nan,
                "fit_sec": np.nan,
                "note": "Skipped (only one class present)",
            }

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
        )

        # Fit
        t0 = time.time()
        clf = fit_logreg(X_train, y_train, c=c, tol=hp["tol"], max_iter=hp["max_iter"], class_weight=class_weight)
        fit_sec = time.time() - t0

        # Evaluate
        train_acc = clf.score(X_train, y_train)
        test_acc = clf.score(X_test, y_test)

        # Count nonzero features
        nnz = count_nnz_features(clf)

        return {
            "file": file_name,
            "target": target_option,
            "C": c,
            "max_iter": hp["max_iter"],
            "tol": hp["tol"],
            "class_weight": class_weight if class_weight is not None else "none",
            "n_samples_used": n_used,
            "n_rows_dropped": n_dropped,
            "n_features": D,
            "n_classes": len(classes),
            "train_acc": float(train_acc),
            "test_acc": float(test_acc),
            "nnz_features": int(nnz),
            "fit_sec": float(fit_sec),
            "note": "",
        }

    except Exception as e:
        return {
            "file": file_name,
            "target": target_option,
            "C": c,
            "max_iter": hp.get("max_iter"),
            "tol": hp.get("tol"),
            "class_weight": class_weight if class_weight is not None else "none",
            "n_samples_used": np.nan,
            "n_rows_dropped": np.nan,
            "n_features": np.nan,
            "n_classes": np.nan,
            "train_acc": np.nan,
            "test_acc": np.nan,
            "nnz_features": np.nan,
            "fit_sec": np.nan,
            "note": f"ERROR: {type(e).__name__}: {e}",
        }

# ---------- Main (parallel) ----------
def main():
    paths = sorted(glob.glob(PARQUET_GLOB))
    if not paths:
        print(f"No Parquet files found for pattern: {PARQUET_GLOB}")
        return

    # Build task list: files × targets × HP × class_weight
    tasks = []
    for p in paths:
        for target in TARGET_OPTIONS:
            for c in Cs:
                for hp in HP_SCHEDULE:
                    for cw in CLASS_WEIGHTS:
                        tasks.append((p, target, c, hp, cw))
    print(f"Total tasks: {len(tasks)}")

    max_workers = min(os.cpu_count() or 2, 8)  # cap to avoid memory pressure
    print(f"Launching ProcessPoolExecutor with max_workers={max_workers}")

    results = []
    with cf.ProcessPoolExecutor(max_workers=max_workers) as ex:
        for i, res in enumerate(ex.map(run_one_task, tasks, chunksize=1), 1):
            print(f"[{i}/{len(tasks)}] {res['file']} | target={res['target']} | "
                  f"iter={res.get('max_iter')} tol={res.get('tol')} cw={res.get('class_weight')} "
                  f"test_acc={res.get('test_acc')} nnz={res.get('nnz_features')} note={res.get('note')}")
            results.append(res)

    out_df = pd.DataFrame(results)
    out_df.to_csv(RESULT_CSV, index=False, encoding="utf-8")
    print(f"\nAll runs complete. Results saved to: {RESULT_CSV}")

if __name__ == "__main__":
    main()
