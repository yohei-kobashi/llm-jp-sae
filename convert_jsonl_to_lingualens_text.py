#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import json
import random


def _extract_pair(data: dict, target: str) -> tuple[str | None, str | None]:
    output = data.get("output", {})
    input_data = data.get("input", {})

    if "irrealis_remaining" not in output:
        return None, None
    if output["irrealis_remaining"]:
        return None, None

    if target == "suppose":
        acceptable = output.get("S2_suppose_that_acceptable")
        pair_text = output.get("S2_suppose_that")
        if acceptable:
            return input_data.get("text"), pair_text
        return None, None

    acceptable = output.get("S1_modality_removed_acceptable")
    pair_text = output.get("S1_modality_removed")
    modal = input_data.get("Modal_Verb")
    if acceptable and modal == target:
        return input_data.get("text"), pair_text
    return None, None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert JSONL minimal data to text for analyzer_lingualens.py."
    )
    parser.add_argument(
        "--input",
        default="data/minimal_pairs_acceptability.jsonl",
        help="Input JSONL path (default: data/minimal_pairs_acceptability.jsonl)",
    )
    parser.add_argument(
        "--output",
        default="data/minimal_pairs_train.txt",
        help="Output train TXT path (default: data/minimal_pairs_train.txt)",
    )
    parser.add_argument(
        "--test-output",
        default="data/minimal_pairs_test.jsonl",
        help="Output test JSONL path (default: data/minimal_pairs_test.jsonl)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Test split ratio in [0.0, 1.0] (default: 0.1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible train/test split (default: 42)",
    )
    parser.add_argument(
        "--target",
        choices=["will", "can", "could", "may", "might", "must", "shall", "should", "would", "ought to", "suppose"],
        default="will",
        help="Modal verb target to judge",
    )

    args = parser.parse_args()
    if not (0.0 <= args.test_ratio <= 1.0):
        raise ValueError("--test-ratio must be in [0.0, 1.0].")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.test_output) or ".", exist_ok=True)

    selected_rows: list[tuple[str, str, str, dict]] = []
    with open(args.input, "r", encoding="utf-8", newline="") as f_in:
        for row in f_in:
            if not row.strip():
                continue
            data = json.loads(row)
            original, minimal_pair = _extract_pair(data, args.target)
            if original and minimal_pair:
                selected_rows.append((row.rstrip("\n"), original, minimal_pair, data))

    rng = random.Random(args.seed)
    indices = list(range(len(selected_rows)))
    rng.shuffle(indices)
    test_size = int(len(selected_rows) * args.test_ratio)
    test_idx = set(indices[:test_size])

    train_rows = [selected_rows[i] for i in range(len(selected_rows)) if i not in test_idx]
    test_rows = [selected_rows[i] for i in range(len(selected_rows)) if i in test_idx]

    with open(args.output, "w", encoding="utf-8") as f_out:
        for _, original, minimal_pair, _ in train_rows:
            f_out.write(f"{original}\n")
            f_out.write(f"{minimal_pair}\n")

    with open(args.test_output, "w", encoding="utf-8") as f_test:
        for raw, _, _, _ in test_rows:
            f_test.write(raw + "\n")

    print(
        "Done.\n"
        f"Input rows: {len(selected_rows)} (filtered by target/pair rules)\n"
        f"Train rows: {len(train_rows)} -> {args.output}\n"
        f"Test rows: {len(test_rows)} -> {args.test_output}"
    )


if __name__ == "__main__":
    main()
