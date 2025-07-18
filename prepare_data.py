import os
import json
import gzip
import math
import random
import time
from pathlib import Path
from typing import List, Iterator

import requests
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from config import UsrConfig, DataConfig

# -----------------------------------------------------------------------------
# GLOBALS ---------------------------------------------------------------------
# -----------------------------------------------------------------------------

CHUNK_SIZE = 1 << 14  # 16 KiB per stream read
PROGRESS_EVERY = 100  # flush cadence (batches) when tokenising
SEED = 42            # reproducibility

# -----------------------------------------------------------------------------
# RESUMABLE I/O UTILS ---------------------------------------------------------
# -----------------------------------------------------------------------------

def _download_url(url: str, dest: Path, retry: int = 3) -> None:
    """Download *url* → *dest* with progress & resume‑skip."""
    if dest.exists() and dest.stat().st_size > 0:
        return  # already fetched

    dest.parent.mkdir(parents=True, exist_ok=True)
    attempt = 0
    while attempt < retry:
        try:
            with requests.get(url, stream=True, timeout=30) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                tmp = dest.with_suffix(".part")
                with open(tmp, "wb") as f, tqdm(
                    total=total,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=f"Downloading {dest.name}",
                    leave=False,
                ) as bar:
                    for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                        if chunk:
                            f.write(chunk)
                            bar.update(len(chunk))
            os.replace(tmp, dest)
            print(f"  ✓ {dest.name} downloaded")
            return
        except Exception as e:
            attempt += 1
            print(f"  download error ({attempt}/{retry}) for {url}: {e}")
            time.sleep(2 * attempt)
    raise RuntimeError(f"Failed to download {url}")


def _iter_jsonl(path: Path) -> Iterator[str]:
    """Yield the *text* field for every JSON‑line in *path* (handles .gz)."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            try:
                obj = json.loads(line)
                yield obj.get("text") or obj.get("content", "")
            except Exception:
                continue  # skip malformed

# -----------------------------------------------------------------------------
# SAMPLING WITH CACHE ---------------------------------------------------------
# -----------------------------------------------------------------------------

def _load_or_sample(lines: List[str], rate: float, cache_path: Path) -> List[str]:
    """Sample *rate* proportion of *lines* (line‑level), caching result."""
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            cached = [l.rstrip("\n") for l in f]
        print(f"  ✓ loaded cached sample ({len(cached)}/{len(lines)}) from {cache_path.name}")
        return cached

    sample_size = max(1, int(len(lines) * rate))
    print(f"  Sampling {sample_size}/{len(lines)} lines …")
    sampled = random.sample(lines, sample_size)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        for t in sampled:
            f.write(t.replace("\n", " ") + "\n")
    print(f"  ✓ sample saved → {cache_path.name}")
    return sampled

# -----------------------------------------------------------------------------
# TOKENISATION HELPERS --------------------------------------------------------
# -----------------------------------------------------------------------------

def batch_tokenize(tokenizer: AutoTokenizer, texts: List[str], max_length: int, pad_id: int) -> torch.Tensor:
    ids = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )["input_ids"]
    return ids[ids.ne(pad_id).all(dim=1)]


def _token_file(name: str, save_dir: Path) -> Path:
    return save_dir / f"tokenized_{name}.pt"


def tokenize_corpus(
    name: str,
    tokenizer: AutoTokenizer,
    corpus: List[str],
    seq_len: int,
    pad_id: int,
    batch_size: int,
    save_dir: Path,
) -> torch.Tensor:
    """Tokenise *corpus* (list of strings) with resumability."""

    out_path = _token_file(name, save_dir)
    if out_path.exists():
        print(f"  ✓ tokens exist for {name} — skipping tokenisation")
        return torch.load(out_path)

    per_doc = seq_len + 1
    n_docs = len(corpus)
    buf = torch.zeros((n_docs, per_doc), dtype=torch.int32)

    processed = 0
    pbar = tqdm(range(0, n_docs, batch_size), desc=f"Tokenising {name}")
    for start in pbar:
        end = min(start + batch_size, n_docs)
        tok = batch_tokenize(tokenizer, corpus[start:end], per_doc, pad_id)
        buf[processed : processed + tok.size(0)] = tok
        processed += tok.size(0)
        pbar.set_postfix({"docs": processed, "%": f"{processed/n_docs:.1%}"})

        if processed % (batch_size * PROGRESS_EVERY) == 0:
            torch.save(buf[:processed].clone(), out_path.with_suffix(".part"))

    torch.save(buf[:processed], out_path)
    out_path.with_suffix(".part").unlink(missing_ok=True)
    print(f"  ✓ saved tokens → {out_path.name}")
    return buf[:processed]

# -----------------------------------------------------------------------------
# DATA PREPARATION ------------------------------------------------------------
# -----------------------------------------------------------------------------

def prepare_dolma(tmp_dir: Path, rate: float) -> List[str]:
    """Download dolma json.gz shards & sample lines."""
    url_list = "https://huggingface.co/datasets/allenai/dolma/raw/main/urls/v1_6-sample.txt"
    url_file = tmp_dir / "dolma_urls.txt"
    _download_url(url_list, url_file)

    with open(url_file, "r", encoding="utf-8") as f:
        shards = [u.strip() for u in f if u.strip()]

    shard_dir = tmp_dir / "dolma_shards"
    texts: List[str] = []
    for url in tqdm(shards, desc="Processing dolma shards", unit="file"):
        dest = shard_dir / Path(url).name
        _download_url(url, dest)
        # iterate each JSON line as separate doc
        for txt in _iter_jsonl(dest):
            if txt:
                texts.append(txt)
    print(f"  collected {len(texts)} total dolma lines")
    return _load_or_sample(texts, rate, tmp_dir / "dolma_sample.txt")


def prepare_ja_warp_html(tmp_dir: Path, rate: float) -> List[str]:
    base = "https://gitlab.llm-jp.nii.ac.jp/datasets/llm-jp-corpus-v3/-/raw/main/ja/ja_warp_html/level0"
    shard_names = [
        "html_nii_01-06.jsonl.gz",
        "html_nii_07-12.jsonl.gz",
    ]
    texts: List[str] = []
    for name in shard_names:
        url = f"{base}/{name}"
        dest = tmp_dir / name
        _download_url(url, dest)
        texts.extend(t for t in _iter_jsonl(dest) if t)
    print(f"  collected {len(texts)} warp_html lines")
    return _load_or_sample(texts, rate, tmp_dir / "warp_sample.txt")

# -----------------------------------------------------------------------------
# MAIN ------------------------------------------------------------------------
# -----------------------------------------------------------------------------

def main() -> None:
    # RNG reproducibility
    random.seed(SEED)
    torch.manual_seed(SEED)

    usr_cfg = UsrConfig()
    data_cfg = DataConfig()

    tmp_dir = Path(usr_cfg.raw_data_dir) / "tmp_download"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    save_dir = Path(usr_cfg.tokenized_data_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1) PREPARE TEXT CORPORA ------------------------------------------------
    print("▶ Preparing dolma sample …")
    dolma_texts = prepare_dolma(tmp_dir, data_cfg.dolma_sample_rate)
    print(f"  dolma sample: {len(dolma_texts)} lines")

    print("▶ Preparing ja_warp_html sample …")
    warp_texts = prepare_ja_warp_html(tmp_dir, data_cfg.warp_sample_rate)
    print(f"  warp_html sample: {len(warp_texts)} lines")

    # 2) TOKENISER -----------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(usr_cfg.model_name_or_dir)
    if tokenizer.pad_token is None:
        raise ValueError("Tokenizer has no pad token.")
    pad_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

    # 3) TOKENISATION (resumable) -------------------------------------------
    dolma_tok = tokenize_corpus(
        "dolma", tokenizer, dolma_texts, data_cfg.seq_len, pad_id, data_cfg.batch_size_tokenizer, save_dir
    )
    warp_tok = tokenize_corpus(
        "warp_html", tokenizer, warp_texts, data_cfg.seq_len, pad_id, data_cfg.batch_size_tokenizer, save_dir
    )

    # 4) COMBINE & SPLIT -----------------------------------------------------
    combo_path = save_dir / "combined.pt"
    if combo_path.exists():
        print("  ✓ combined.pt exists — skipping combine/shuffle")
        combined = torch.load(combo_path)
    else:
        combined = torch.cat([dolma_tok, warp_tok], dim=0)
        combined = combined[torch.randperm(combined.size(0))]
        torch.save(combined, combo_path)
        print("  ✓ saved shuffled combined dataset")

    ratios = data_cfg.train_val_test_ratio
    n_total = combined.size(0)
    n_train = math.floor(n_total * ratios[0])
    n_val = math.floor(n_total * ratios[1])

    splits = {
        "train_data.pt": combined[:n_train],
        "val_data.pt": combined[n_train : n_train + n_val],
        "test_data.pt": combined[n_train + n_val :],
    }

    for fname, tensor in splits.items():
        path = save_dir / fname
