import os
import io
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
# RESUMABLE UTILS (with progress bars) ----------------------------------------
# -----------------------------------------------------------------------------

CHUNK_SIZE = 1 << 14  # 16 KiB
PROGRESS_EVERY = 100  # batches before flushing temp token file


def _download_url(url: str, dest: Path, retry: int = 3) -> None:
    """Download *url* to *dest* streaming with progress. Skip if already exists."""
    if dest.exists() and dest.stat().st_size > 0:
        return  # already downloaded

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
    """Yield text fields from (possibly gzipped) JSONL *path*. Shows progress."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            try:
                yield json.loads(line)["text"]
            except Exception:
                continue  # skip malformed lines

# -----------------------------------------------------------------------------
# SAMPLING WITH REPLAY ---------------------------------------------------------
# -----------------------------------------------------------------------------

def _load_or_sample(lines: List[str], rate: float, cache_path: Path) -> List[str]:
    """Return a random *rate* subset, caching to *cache_path* for reproducibility."""
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            cached = [l.rstrip("\n") for l in f]
        print(f"  ✓ loaded cached sample ({len(cached)}/{len(lines)}) from {cache_path.name}")
        return cached

    sample_size = max(1, int(len(lines) * rate))
    print(f"  Sampling {sample_size}/{len(lines)} documents …")
    sampled = random.sample(lines, sample_size)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        for t in sampled:
            f.write(t.replace("\n", " ") + "\n")
    print(f"  ✓ sample saved → {cache_path.name}")
    return sampled

# -----------------------------------------------------------------------------
# TOKENIZATION HELPERS (RESUMABLE) --------------------------------------------
# -----------------------------------------------------------------------------

def batch_tokenize(tokenizer: AutoTokenizer, texts: List[str], max_length: int, pad_id: int) -> torch.Tensor:
    out = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )["input_ids"]
    return out[out.ne(pad_id).all(dim=1)]


def _token_file(corpus_name: str, save_dir: Path) -> Path:
    return save_dir / f"tokenized_{corpus_name}.pt"


def tokenize_corpus(
    corpus_name: str,
    tokenizer: AutoTokenizer,
    corpus: List[str],
    seq_len: int,
    pad_id: int,
    batch_size: int,
    save_dir: Path,
) -> torch.Tensor:
    """Tokenize *corpus* with resumability & progress bars."""

    out_path = _token_file(corpus_name, save_dir)
    if out_path.exists():
        print(f"  ✓ tokens exist for {corpus_name} — skipping tokenization")
        return torch.load(out_path)

    tokens_per_doc = seq_len + 1
    num_docs = len(corpus)
    token_buf = torch.zeros((num_docs, tokens_per_doc), dtype=torch.int32)

    processed = 0
    pbar = tqdm(
        range(0, num_docs, batch_size),
        desc=f"Tokenizing {corpus_name}",
        unit="batch",
    )
    for start in pbar:
        end = min(start + batch_size, num_docs)
        batch = corpus[start:end]
        tok = batch_tokenize(tokenizer, batch, tokens_per_doc, pad_id)
        token_buf[processed : processed + tok.size(0)] = tok
        processed += tok.size(0)
        pbar.set_postfix({"docs": processed, "done": f"{processed/num_docs:.1%}"})

        if processed % (batch_size * PROGRESS_EVERY) == 0:
            torch.save(token_buf[:processed].clone(), out_path.with_suffix(".part"))

    torch.save(token_buf[:processed], out_path)
    out_path.with_suffix(".part").unlink(missing_ok=True)
    print(f"  ✓ saved tokens → {out_path.name}")
    return token_buf[:processed]

# -----------------------------------------------------------------------------
# DATA PREPARATION FUNCTIONS ---------------------------------------------------
# -----------------------------------------------------------------------------

def prepare_dolma(tmp_dir: Path, rate: float) -> List[str]:
    url_list_url = "https://huggingface.co/datasets/allenai/dolma/raw/main/urls/v1_6-sample.txt"
    url_list_path = tmp_dir / "dolma_urls.txt"
    _download_url(url_list_url, url_list_path)

    with open(url_list_path, "r", encoding="utf-8") as f:
        urls = [u.strip() for u in f if u.strip()]

    page_dir = tmp_dir / "dolma_pages"
    page_dir.mkdir(exist_ok=True)

    texts: List[str] = []
    for url in tqdm(urls, desc="Fetching dolma pages", unit="url"):
        page_path = page_dir / (url.replace("/", "_"))
        try:
            _download_url(url, page_path)
            with open(page_path, "r", encoding="utf-8", errors="ignore") as fp:
                texts.append(fp.read())
        except Exception:
            continue

    sampled = _load_or_sample(texts, rate, tmp_dir / "dolma_sample.txt")
    return sampled


def prepare_ja_warp_html(tmp_dir: Path, rate: float) -> List[str]:
    base = "https://gitlab.llm-jp.nii.ac.jp/datasets/llm-jp-corpus-v3/-/raw/main/ja/ja_warp_html/level0"
    files = [
        "html_nii_01-06.jsonl.gz",
        "html_nii_07-12.jsonl.gz",
    ]
    texts: List[str] = []
    for fname in files:
        url = f"{base}/{fname}"
        dest = tmp_dir / fname
        _download_url(url, dest)
        texts.extend(_iter_jsonl(dest))

    sampled = _load_or_sample(texts, rate, tmp_dir / "warp_sample.txt")
    return sampled

# -----------------------------------------------------------------------------
# MAIN -------------------------------------------------------------------------
# -----------------------------------------------------------------------------

def main() -> None:
    random.seed(42)
    usr_cfg = UsrConfig()
    data_cfg = DataConfig()
    print(usr_cfg)
    print(data_cfg)

    tmp_dir = Path(usr_cfg.raw_data_dir) / "tmp_download"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    save_dir = Path(usr_cfg.tokenized_data_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------
    # PREPARE CORPORA ----------------------------------------------
    # ---------------------------------------------------------------
    print("▶ Preparing dolma sample …")
    dolma_texts = prepare_dolma(tmp_dir, data_cfg.dolma_sample_rate)
    print(f"  dolma sample: {len(dolma_texts)} docs")

    print("▶ Preparing ja_warp_html sample …")
    warp_texts = prepare_ja_warp_html(tmp_dir, data_cfg.warp_sample_rate)
    print(f"  warp_html sample: {len(warp_texts)} docs")

    # ---------------------------------------------------------------
    # TOKENIZER -----------------------------------------------------
    # ---------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(usr_cfg.model_name_or_dir)
    if tokenizer.pad_token is None:
        raise ValueError("Tokenizer has no pad token.")
    pad_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

    # ---------------------------------------------------------------
    # TOKENIZATION (resumable) --------------------------------------
    # ---------------------------------------------------------------
    dolma_tok = tokenize_corpus(
        "dolma",
        tokenizer,
        dolma_texts,
        data_cfg.seq_len,
        pad_id,
        data_cfg.batch_size_tokenizer,
        save_dir,
    )
    warp_tok = tokenize_corpus(
        "warp_html",
        tokenizer,
        warp_texts,
        data_cfg.seq_len,
        pad_id,
        data_cfg.batch_size_tokenizer,
        save_dir,
    )

    # ---------------------------------------------------------------
    # COMBINE & SPLIT ----------------------------------------------
    # ---------------------------------------------------------------
    combined_path = save_dir / "combined.pt"
    if combined_path.exists():
        print("  ✓ combined.pt exists — skipping combine/shuffle")
        combined = torch.load(combined_path)
    else:
        combined = torch.cat([dolma_tok, warp_tok], dim=0)
        combined = combined[torch.randperm(combined.size(0))]
        torch.save(combined, combined_path)
        print("  ✓ saved shuffled combined dataset")

    ratios = data_cfg.train_val
