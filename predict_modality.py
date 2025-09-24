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
C_FIXED = 1.0
SOLVER = "saga"
PENALTY = "l1"
N_JOBS = 1                     # Important: 1 per process; parallelism is at process level
MULTI_CLASS = "auto"           # Quiet deprecation; >=1.7 will default to multinomial

# Minimal hyperparameter schedule
HP_SCHEDULE = [
    {"max_iter": 1000, "tol": 1e-3},
    {"max_iter": 2000, "tol": 1e-4},
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
        pal_ok = pal.str.lower().str_
