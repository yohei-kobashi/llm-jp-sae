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

# =============================================================================
# GLOBAL CONSTANTS
# =============================================================================
CHUNK_SIZE = 1 << 14       # 16 KiB per streamed read
PROGRESS_EVERY = 100       # save partial tokens every N batches
SEED = 42                  # RNG seed for reproducibility

# =============================================================================
# DOWNLOAD HELPER (RESUMABLE, SSL‑FALLBACK & PROGRESS)
# =============================================================================

def _download_url(url: str, dest: Path, retry: int = 3) -> None:
    """Stream *url* to *dest* with resumption awareness.

    * Skips if *dest* already exists and is non‑empty.
    * Uses `tqdm` progress bar when `Content‑Length` header present.
    * First attempts secure TLS. On certificate failure, retries with
      `verify=False` (insecure) while issuing a warning.
    * Writes into `*.part` temporary file and atomically renames on success.
    """
    if dest.exists() and dest.stat().st_size > 0:
        return  # already downloaded

    dest.parent.mkdir(parents=True, exist_ok=True)

    verify = False  # start with secure TLS
    attempt = 0
    while attempt < retry:
        try:
            with requests.get(url, stream=True, timeout=30, verify=verify) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0) or 0)
                tmp = dest.with_suffix(".part")
                with open(tmp, "wb") as f, tqdm(
                    total=total,
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024,
                    desc=f"↯ {dest.name}",
                    leave=False,
                ) as bar:
                    for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                        if chunk:
                            f.write(chunk)
                            bar.update(len(chunk))
            os.replace(tmp, dest)
            print(f"  ✓ downloaded {dest.name}")
            return
        except requests.exceptions.SSLError as e:
            if verify:
                import urllib3, warnings
                warnings.warn(
                    f"SSL verification failed for {url}. Retrying with verify=False.",
                    RuntimeWarning,
                )
                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
                verify = False
                continue  # retry same attempt
            attempt += 1
            print(f"  SSL error ({attempt}/{retry}) for {url}: {e}")
        except Exception as e:
            attempt += 1
            print(f"  download error ({attempt}/{retry}) for {url}: {e}")
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Failed to download {url}")

# =============================================================================
# FILE ITERATORS
# =============================================================================

def _iter_jsonl(path: Path) -> Iterator[str]:
    """Yield text/content field from JSONL (supports *.gz)."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            try:
                obj = json.loads(line)
                yield obj.get("text") or obj.get("content", "")
            except Exception:
                continue  # malformed line

# =============================================================================
# SAMPLING WITH CACHE
# =============================================================================

def _load_or_sample(lines: List[str], rate: float, cache_path: Path) -> List[str]:
    """Return cached sample if available else random sample of *rate* lines."""
    if cache_path.exists() and cache_path.stat().st_size > 0:
        with open(cache_path, "r", encoding="utf-8") as f:
            sample = [l.rstrip("\n") for l in f]
        print(f"  ✓ loaded cached sample ({len(sample)} lines) from {cache_path.name}")
        return sample

    sample_size = max(1, int(len(lines) * rate))
    print(f"  Sampling {sample_size}/{len(lines)} lines …")
    sample = random.sample(lines, sample_size)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        for t in sample:
            f.write(t.replace("\n", " ") + "\n")
    print(f"  ✓ sample saved → {cache_path.name}")
    return sample

# =============================================================================
# TOKENISATION HELPERS
# =============================================================================

def _batch_tokenize(tokenizer: AutoTokenizer, texts: List[str], max_len: int, pad_id: int) -> torch.Tensor:
    ids = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
    )["input_ids"]
    return ids[ids.ne(pad_id).all(dim=1)]


def _token_file(name: str, out_dir: Path) -> Path:
    return out_dir / f"tokenized_{name}.pt"


def tokenize_corpus(
    name: str,
    tokenizer: AutoTokenizer,
    corpus: List[str],
    seq_len: int,
    pad_id: int,
    batch_size: int,
    out_dir: Path,
) -> torch.Tensor:
    """Tokenise *corpus* with resumability and progress bars."""

    out_path = _token_file(name, out_dir)
    if out_path.exists() and out_path.stat().st_size > 0:
        print(f"  ✓ tokens exist for {name} — loading from disk")
        return torch.load(out_path)

    per_doc = seq_len + 1
    n_docs = len(corpus)
    buf = torch.zeros((n_docs, per_doc), dtype=torch.int32)

    processed = 0
    pbar = tqdm(range(0, n_docs, batch_size), desc=f"Tokenising {name}")
    for start in pbar:
        end = min(start + batch_size, n_docs)
        tok = _batch_tokenize(tokenizer, corpus[start:end], per_doc, pad_id)
        buf[processed : processed + tok.size(0)] = tok
        processed += tok.size(0)
        pbar.set_postfix({"docs": processed, "%": f"{processed/n_docs:.1%}"})

        if processed % (batch_size * PROGRESS_EVERY) == 0:
            torch.save(buf[:processed].clone(), out_path.with_suffix(".part"))

    torch.save(buf[:processed], out_path)
    out_path.with_suffix(".part").unlink(missing_ok=True)
    print(f"  ✓ saved tokens → {out_path.name}")
    return buf[:processed]

# =============================================================================
# CORPUS PREPARATION
# =============================================================================

def prepare_dolma(tmp_dir: Path, rate: float) -> List[str]:
    """Return dolma sample lines (cached if exists)."""
    sample_cache = tmp_dir / "dolma_sample.txt"
    if sample_cache.exists() and sample_cache.stat().st_size > 0:
        print("  ✓ dolma_sample.txt found — skipping dolma shard processing")
        with open(sample_cache, "r", encoding="utf-8") as f:
            return [l.rstrip("\n") for l in f]

    # Build sample ------------------------------------------------------
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
        for txt in _iter_jsonl(dest):
            if txt:
                texts.append(txt)
    print(f"  collected {len(texts)} total dolma lines")
    return _load_or_sample(texts, rate, sample_cache)


def prepare_ja_warp_html(tmp_dir: Path, rate: float) -> List[str]:
    """Return warp_html sample lines (cached if exists)."""
    sample_cache = tmp_dir / "warp_sample.txt"
    if sample_cache.exists() and sample_cache.stat().st_size > 0:
        print("  ✓ warp_sample.txt found — skipping warp_html download")
        with open(sample_cache, "r", encoding="utf-8") as f:
            return [l.rstrip("\n") for l in f]

    base = "https://gitlab.llm-jp.nii.ac.jp/datasets/llm-jp-corpus-v3/-/raw/main/ja/ja_warp_html/level0"
    shard_names = ["html_nii_01-06.jsonl.gz", "html_nii_07-12.jsonl.gz"]

    texts: List[str] = []
    for name in shard_names:
        url = f"{base}/{name}"
        dest = tmp_dir / name
        _download_url(url, dest)
        texts.extend(t for t in _iter_jsonl(dest) if t)
    print(f"  collected {len(texts)} warp_html lines")
    return _load_or_sample(texts, rate, sample_cache)

# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main() -> None:
    random.seed(SEED)
    torch.manual_seed(SEED)

    usr_cfg = UsrConfig()
    cfg = DataConfig()

    tmp_dir = Path(usr_cfg.raw_data_dir) / "tmp_download"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    out_dir = Path(usr_cfg.tokenized_data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # PREPARE CORPORA
    # ------------------------------------------------------------------
    print("▶ Preparing dolma sample …")
    dolma_lines = prepare_dolma(tmp_dir, rate=cfg.dolma_sample_rate)
    print(f"  dolma sample: {len(dolma_lines)} lines")

    print("▶ Preparing ja_warp_html sample …")
    warp_lines = prepare_ja_warp_html(tmp_dir, rate=cfg.warp_sample_rate)
    print(f"  warp_html sample: {len(warp_lines)} lines")

    # ------------------------------------------------------------------
    # TOKENISER INIT
    # ------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(usr_cfg.model_name_or_dir)
    if tokenizer.pad_token is None:
        raise ValueError("Tokenizer lacks a pad token")
    pad_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

    # ------------------------------------------------------------------
    # TOKENISATION (resumable)
    # ------------------------------------------------------------------
    dolma_tok = tokenize_corpus(
        "dolma",
        tokenizer,
        dolma_lines,
        cfg.seq_len,
        pad_id,
        cfg.batch_size_tokenizer,
        out_dir,
    )
    warp_tok = tokenize_corpus(
        "warp_html",
        tokenizer,
        warp_lines,
        cfg.seq_len,
        pad_id,
        cfg.batch_size_tokenizer,
        out_dir,
    )

    # ------------------------------------------------------------------
    # COMBINE, SHUFFLE, SPLIT
    # ------------------------------------------------------------------
    combined_path = out_dir / "combined.pt"
    if combined_path.exists() and combined_path.stat().
