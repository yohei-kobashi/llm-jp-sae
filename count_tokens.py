#!/usr/bin/env python
import argparse
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoTokenizer

from config import UsrConfig, DataConfig  # 既存設定を再利用

def load_tensor(path: Path) -> torch.Tensor:
    return torch.load(path, map_location="cpu")

def count_stats(t: torch.Tensor, pad_id: int) -> Dict[str, float]:
    # t: (num_docs, seq_len+1)
    nonpad_mask = (t != pad_id)
    total_tokens = int(nonpad_mask.sum().item())
    num_docs = t.size(0)
    seq_len = t.size(1)
    # 1ドキュメントあたり非pad平均
    avg_len = total_tokens / num_docs if num_docs else 0.0
    pad_tokens = num_docs * seq_len - total_tokens
    pad_ratio = pad_tokens / (num_docs * seq_len) if num_docs else 0.0
    return {
        "docs": num_docs,
        "seq_len": seq_len,
        "total_tokens_nonpad": total_tokens,
        "avg_len_nonpad": avg_len,
        "pad_ratio": pad_ratio,
    }

def format_stats(name: str, stats: Dict[str, float]) -> str:
    return (
        f"{name:18s} | docs={stats['docs']:>7d} | seq={stats['seq_len']:>4d} | "
        f"tokens(nonpad)={stats['total_tokens_nonpad']:,} | "
        f"avg_nonpad={stats['avg_len_nonpad']:.1f} | pad_ratio={stats['pad_ratio']*100:.2f}%"
    )

def main():
    parser = argparse.ArgumentParser(description="Count token stats from saved .pt tensors.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Directory containing tokenized_*.pt / train_data.pt (default = UsrConfig.tokenized_data_dir)",
    )
    parser.add_argument(
        "--show-all",
        action="store_true",
        help="Also show full combined and per-corpus tokenized files if present."
    )
    args = parser.parse_args()

    usr_cfg = UsrConfig()
    data_cfg = DataConfig()
    data_dir = args.data_dir or Path(usr_cfg.tokenized_data_dir)

    if not data_dir.exists():
        raise SystemExit(f"Data dir not found: {data_dir}")

    # tokenizer for pad id
    tokenizer = AutoTokenizer.from_pretrained(usr_cfg.model_name_or_dir)
    if tokenizer.pad_token is None:
        raise ValueError("Tokenizer has no pad token.")
    pad_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

    # candidate files
    core_files: List[str] = ["train_data.pt", "val_data.pt", "test_data.pt"]
    extra_files: List[str] = [
        "tokenized_dolma.pt",
        "tokenized_warp_html.pt",
        "combined.pt",
    ]

    targets: List[str] = []
    for f in core_files:
        if (data_dir / f).exists():
            targets.append(f)
    if args.show_all:
        for f in extra_files:
            if (data_dir / f).exists():
                targets.append(f)

    if not targets:
        raise SystemExit("No target .pt files found.")

    print(f"# Token stats in {data_dir}\n")
    grand_total = 0
    for fname in targets:
        tensor = load_tensor(data_dir / fname)
        stats = count_stats(tensor, pad_id)
        grand_total += stats["total_tokens_nonpad"]
        print(format_stats(fname, stats))

    print(f"\nGrand total non-pad tokens: {grand_total:,}")

if __name__ == "__main__":
    main()
