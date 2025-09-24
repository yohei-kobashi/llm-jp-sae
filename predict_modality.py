"""
Usage
-----
1) Put your Parquet files under:  modality_data/
   Required columns in each file:
       - "Modal_Verb"                 (string)
       - "Palmer_Expected"            (string, may contain "unknown")
       - "Quirk_Expected"             (string, may contain "unknown")
       - "sae_tokens"                 (list or ndarray of strings)
       - "sae_latent_indices"         (list or ndarray of list/ndarray[int] per token)
       - "sae_latent_acts"            (list or ndarray of list/ndarray[float] per token)

2) This script runs ALL target options in one execution:
       - "modal_verb"
       - "palmer_expected"   (drops rows where Palmer_Expected == "unknown", case-insensitive)
       - "quirk_expected"    (drops rows where Quirk_Expected == "unknown", case-insensitive)
       - "palmer_quirk_pair" (drops rows where either is "unknown"; label is "Palmer|Quirk")

3) Minimal hyperparameter tuning:
   - C is fixed to 1.0.
   - Two (max_iter, tol) pairs are tried: (200, 1e-3) and (500, 1e-4).
   - class_weight is tried with and without "balanced".
   The Cartesian product of the above is evaluated for each (file, target).

4) Parallel execution:
   - Parallelizes across (file × target × hp × class_weight) with ProcessPoolExecutor.
   - Prevents thread oversubscription inside each process via environment variables.
   - Sets scikit-learn's n_jobs to 1 inside each process.

5) Run:
       python predict_modality_parallel.py

6) Output:
       - Combined CSV with one row per (file, target, hp, class_weight) at RESULT_CSV.
"""

import os
# Prevent thread oversubscription inside each process (MKL/OpenBLAS/NumExpr/OpenMP)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import glob
import time
import concurrent.futures as cf
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
RESULT_CSV = "results_modality_logreg_all_targets_min_tune_parallel.csv"

# Train/test split
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Feature pooling
USE_MODAL_TOKEN_ONLY = False   # True: only use tokens equal to the current "Modal_Verb"

# Logistic regression core settings
C_FIXED = 1.0                  # Fixed as requested
SOLVER = "saga"
PENALTY = "l1"
N_JOBS = 1                     # IMPORTANT: 1 per process; avoid process × thread oversubscription
MULTI_CLASS = "auto"           # avoid deprecation warning; >=1.7 will be multinomial

# Minimal hyperparameter schedule
HP_SCHEDULE = [
    {"max_iter": 1000, "tol": 1e-3},
    {"max_iter": 2000, "tol": 1e-4},
]
CLASS_WEIGHTS = [None, "balanced"]   # try with/without balancing

# All target options to run in a single execution
TARGET_OPTIONS = [
    "modal_verb",
    "palmer_expected",
    "quirk_expected",
    "palmer_quirk_pair",
]
UNKNOWN_TOKEN = "unknown"      # case-insensitive match


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


def normalize_unknown(s):
    """Return True if s equals UNKNOWN_TOKEN (case-insensitive); handles None safely."""
    if s is None:
        return True
    try:
        return str(s).strip().lower() == UNKNOWN_TOKEN
    except Exception:
        return False


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


def select_and_encode_labels(df, target_option):
    """
    Build a filtered DataFrame and a string label series based on `target_option`.
    After removing 'unknown' tokens, categories with fewer than 5 samples are also dropped.
    Returns: filtered_df, label_str_series, dropped_count, option_used
    """
    option = target_option.lower().strip()

    if option == "modal_verb":
        # Use Modal_Verb as-is (no unknown filtering).
        label = df["Modal_Verb"].astype(str)
        filt = pd.Series([True] * len(df), index=df.index)

    elif option == "palmer_expected":
        # Drop rows where Palmer_Expected == "unknown"
        pal = df["Palmer_Expected"].astype(str)
        keep = pal.str.lower().str.strip() != UNKNOWN_TOKEN
        filt = keep.fillna(False)
        label = pal[filt]

    elif option == "quirk_expected":
        # Drop rows where Quirk_Expected == "unknown"
        quirk = df["Quirk_Expected"].astype(str)
        keep = quirk.str.lower().str.strip() != UNKNOWN_TOKEN
        filt = keep.fillna(False)
        label = quirk[filt]

    elif option == "palmer_quirk_pair":
        # Drop rows where either is "unknown"; label is "Palmer|Quirk"
        pal = df["Palmer_Expected"].astype(str)
        quirk = df["Quirk_Expected"].astype(str)
        pal_ok = pal.str.lower().str.strip() != UNKNOWN_TOKEN
        quirk_ok = quirk.str.lower().str.strip() != UNKNOWN_TOKEN
        keep = pal_ok & quirk_ok
        filt = keep.fillna(False)
        label = (pal[filt].str.strip() + "|" + quirk[filt].str.strip())

    else:
        raise ValueError(f"Unknown TARGET_OPTION: {target_option}")

    # Step 1: filter out "unknown"
    filtered_df = df[filt].reset_index(drop=True)
    label = label.reset_index(drop=True)
    dropped = int((~filt).sum())

    # Step 2: drop categories with < 5 samples
    counts = label.value_counts()
    valid_classes = counts[counts >= 5].index
    keep_final = label.isin(valid_classes)
    dropped_small = int((~keep_final).sum())

    filtered_df = filtered_df[keep_final].reset_index(drop=True)
    label = label[keep_final].reset_index(drop=True)

    # Update total dropped count
    dropped += dropped_small

    return filtered_df, label, dropped, option


def build_X_y(df, target_option):
    """
    Build the CSR feature matrix X and label vector y from a DataFrame with columns:
      - Modal_Verb, Palmer_Expected, Quirk_Expected, sae_tokens, sae_latent_indices, sae_latent_acts
    Applies unknown filtering depending on target_option.
    Returns: X (csr_matrix), y (np.ndarray), D (int), label_encoder (LabelEncoder),
             n_used (int), n_dropped (int), classes (list[str])
    """
    # Labels with unknown filtering when needed
    df2, label_str, n_dropped, option_used = select_and_encode_labels(df, target_option)

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(label_str.astype(str))

    # Estimate SAE dictionary size
    D = max_idx_from_nested(df2["sae_latent_indices"]) + 1
    if D <= 0:
        raise ValueError("Dictionary size D could not be estimated (no non-empty activations).")

    # Build CSR rows
    rows = []
    if USE_MODAL_TOKEN_ONLY:
        # When using modal tokens only, use surface Modal_Verb to filter tokens
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


def run_one_file_for_target_with_hp(task):
    """
    Worker entry point for a single (file, target, hp, class_weight) combination.
    Returns a result dict with metadata and accuracies.
    """
    path, target_option, hp, class_weight = task
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
                "C": C_FIXED,
                "max_iter": hp["max_iter"],
                "tol": hp["tol"],
                "class_weight": class_weight if class_weight is not None else "none",
                "n_samples_used": n_used,
                "n_rows_dropped": n_dropped,
                "n_features": D,
                "n_classes": len(unique_classes),
                "train_acc": np.nan,
                "test_acc": np.nan,
                "classes": "|".join(map(str, classes)),
                "fit_sec": np.nan,
                "note": "Skipped (only one class present)",
            }

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
        )

        t0 = time.time()
        clf = LogisticRegression(
            solver=SOLVER,
            penalty=PENALTY,
            C=C_FIXED,
            tol=hp["tol"],
            max_iter=hp["max_iter"],
            n_jobs=N_JOBS,
            class_weight=class_weight,
            multi_class=MULTI_CLASS,
            verbose=0,
        )
        clf.fit(X_train, y_train)
        fit_sec = time.time() - t0

        train_acc = clf.score(X_train, y_train)
        test_acc = clf.score(X_test, y_test)

        return {
            "file": file_name,
            "target": target_option,
            "C": C_FIXED,
            "max_iter": hp["max_iter"],
            "tol": hp["tol"],
            "class_weight": class_weight if class_weight is not None else "none",
            "n_samples_used": n_used,
            "n_rows_dropped": n_dropped,
            "n_features": D,
            "n_classes": len(classes),
            "train_acc": float(train_acc),
            "test_acc": float(test_acc),
            "classes": "|".join(map(str, classes)),
            "fit_sec": float(fit_sec),
            "note": "",
        }

    except Exception as e:
        return {
            "file": file_name,
            "target": target_option,
            "C": C_FIXED,
            "max_iter": hp.get("max_iter"),
            "tol": hp.get("tol"),
            "class_weight": class_weight if class_weight is not None else "none",
            "n_samples_used": np.nan,
            "n_rows_dropped": np.nan,
            "n_features": np.nan,
            "n_classes": np.nan,
            "train_acc": np.nan,
            "test_acc": np.nan,
            "classes": "",
            "fit_sec": np.nan,
            "note": f"ERROR: {type(e).__name__}: {e}",
        }


def main():
    """Parallel run over all Parquet files, all targets, and minimal HP grid; save a CSV summary."""
    paths = sorted(glob.glob(PARQUET_GLOB))
    if not paths:
        print(f"No Parquet files found for pattern: {PARQUET_GLOB}")
        return

    # Build the task list (Cartesian product of files × targets × HP × class_weight)
    tasks = []
    for p in paths:
        for target in TARGET_OPTIONS:
            for hp in HP_SCHEDULE:
                for cw in CLASS_WEIGHTS:
                    tasks.append((p, target, hp, cw))
    print(f"Total tasks: {len(tasks)}")

    # Choose worker count carefully (physical cores or half if memory is tight)
    max_workers = min(os.cpu_count() or 2, 20)  # example cap at 8
    print(f"Launching ProcessPoolExecutor with max_workers={max_workers}")

    results = []
    with cf.ProcessPoolExecutor(max_workers=max_workers) as ex:
        # chunksize controls batching; 1 is fine, larger can help if tasks are very short
        for i, res in enumerate(ex.map(run_one_file_for_target_with_hp, tasks, chunksize=1), 1):
            # Lightweight progress log printed in the parent process
            print(f"[{i}/{len(tasks)}] {res['file']} | target={res['target']} | "
                  f"iter={res.get('max_iter')} tol={res.get('tol')} cw={res.get('class_weight')} "
                  f"test_acc={res.get('test_acc')} note={res.get('note')}")
            results.append(res)

    out_df = pd.DataFrame(results)
    out_df.to_csv(RESULT_CSV, index=False, encoding="utf-8")
    print(f"\nAll runs complete. Results saved to: {RESULT_CSV}")


if __name__ == "__main__":
    main()
