#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

DEFAULT_MODELS = [
    "llm-jp-3-1.8b",
    "OLMo-2-0425-1B",
    "Llama-3.2-1B",
    "Qwen2.5-1.5B",
]


def _load_summary(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _layer_sort_key(layer_name: str) -> Tuple[int, str]:
    try:
        return (0, int(layer_name))
    except ValueError:
        return (1, layer_name)


def _share(histogram: Dict[str, int], bucket: str) -> float:
    total = sum(int(value) for value in histogram.values())
    if total == 0:
        return 0.0
    return float(histogram.get(bucket, 0)) / float(total)


def _rank_models(values: Dict[str, float]) -> str:
    ordered = sorted(values.items(), key=lambda item: (-item[1], item[0]))
    groups: List[List[str]] = []
    current_value = None
    for model, value in ordered:
        if current_value is None or value != current_value:
            groups.append([model])
            current_value = value
        else:
            groups[-1].append(model)
    return " > ".join(" = ".join(group) for group in groups)


def _peak_layer_for_bucket(summary: Dict[str, Any], comparison_name: str, bucket: str) -> Dict[str, Any]:
    layer_overlap = summary.get("comparisons", {}).get(comparison_name, {}).get("layer_overlap", {})
    best_layer = None
    best_share = -1.0
    best_count = 0

    for layer_name in sorted(layer_overlap.keys(), key=_layer_sort_key):
        histogram = layer_overlap.get(layer_name, {}).get("overlap_histogram", {})
        share = _share(histogram, bucket)
        count = int(histogram.get(bucket, 0))
        if share > best_share:
            best_layer = layer_name
            best_share = share
            best_count = count

    if best_layer is None:
        return {
            "peak_layer": "",
            "peak_count": 0,
            "peak_share": 0.0,
        }

    return {
        "peak_layer": best_layer,
        "peak_count": best_count,
        "peak_share": best_share,
    }


def _build_rows(
    summaries: Dict[str, Dict[str, Any]],
    models: List[str],
) -> List[Dict[str, Any]]:
    comparison_names = sorted(
        {
            comparison_name
            for summary in summaries.values()
            for comparison_name in summary.get("comparisons", {})
        }
    )

    rows: List[Dict[str, Any]] = []
    for comparison_name in comparison_names:
        buckets = sorted(
            {
                bucket
                for summary in summaries.values()
                for layer_overlap in summary.get("comparisons", {})
                .get(comparison_name, {})
                .get("layer_overlap", {})
                .values()
                for bucket in layer_overlap.get("overlap_histogram", {})
            },
            key=lambda bucket: int(bucket),
        )
        for bucket in buckets:
            peak_counts = {}
            peak_shares = {}
            row: Dict[str, Any] = {
                "comparison": comparison_name,
                "histogram_bucket": bucket,
            }
            for model in models:
                peak = _peak_layer_for_bucket(summaries.get(model, {}), comparison_name, bucket)
                peak_counts[model] = int(peak["peak_count"])
                peak_shares[model] = float(peak["peak_share"])
                row[f"{model}_peak_layer"] = peak["peak_layer"]
                row[f"{model}_peak_count"] = int(peak["peak_count"])
                row[f"{model}_peak_share"] = round(float(peak["peak_share"]), 6)
            row["peak_count_rank"] = _rank_models(peak_counts)
            row["peak_share_rank"] = _rank_models(peak_shares)
            rows.append(row)
    return rows


def _write_tsv(rows: List[Dict[str, Any]], models: List[str], output_path: Path) -> None:
    fieldnames = ["comparison", "histogram_bucket"]
    for model in models:
        fieldnames.append(f"{model}_peak_layer")
        fieldnames.append(f"{model}_peak_count")
        fieldnames.append(f"{model}_peak_share")
    fieldnames.extend(["peak_count_rank", "peak_share_rank"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="	")
        writer.writeheader()
        writer.writerows(rows)


def _print_rows(rows: List[Dict[str, Any]], models: List[str]) -> None:
    header = ["comparison", "bucket"]
    for model in models:
        header.append(f"{model}:peak_layer")
        header.append(f"{model}:peak_count")
        header.append(f"{model}:peak_share")
    header.extend(["peak_count_rank", "peak_share_rank"])
    print("	".join(header))
    for row in rows:
        values = [row["comparison"], str(row["histogram_bucket"])]
        for model in models:
            values.append(str(row[f"{model}_peak_layer"]))
            values.append(str(row[f"{model}_peak_count"]))
            values.append(f"{row[f'{model}_peak_share']:.6f}")
        values.append(row["peak_count_rank"])
        values.append(row["peak_share_rank"])
        print("	".join(values))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare peak overlap_histogram shares across multiple "
            "modal_layer_overlap_summary.json files."
        )
    )
    parser.add_argument(
        "--outputs-dir",
        default="outputs",
        help="Root directory containing per-model modal_layer_overlap_summary.json files.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Model directory names under --outputs-dir.",
    )
    parser.add_argument(
        "--summary-name",
        default="modal_layer_overlap_summary.json",
        help="Summary filename to read from each model directory.",
    )
    parser.add_argument(
        "--output-tsv",
        default="outputs/modal_layer_overlap_histogram_peak_comparison.tsv",
        help="Path to save the peak-layer comparison table as TSV.",
    )
    parser.add_argument(
        "--print",
        action="store_true",
        help="Also print the comparison table to stdout.",
    )
    args = parser.parse_args()

    outputs_dir = Path(args.outputs_dir)
    summaries = {}
    for model in args.models:
        summary_path = outputs_dir / model / args.summary_name
        if not summary_path.exists():
            raise FileNotFoundError(f"Summary file not found: {summary_path}")
        summaries[model] = _load_summary(summary_path)

    rows = _build_rows(summaries, args.models)
    output_path = Path(args.output_tsv)
    _write_tsv(rows, args.models, output_path)
    print(f"Saved histogram peak comparison table: {output_path}")

    if args.print:
        _print_rows(rows, args.models)


if __name__ == "__main__":
    main()
