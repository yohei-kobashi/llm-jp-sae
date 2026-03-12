#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from glob import glob
from typing import Any, Dict, List, Tuple

from LinguaLens.lingualens.utils import save_json_results


def _collect_layer_json_paths(input_dir: str) -> List[str]:
    paths = sorted(glob(os.path.join(input_dir, "*_layer*_evolution.json")))
    if not paths:
        raise FileNotFoundError(
            f"No per-layer crosslayer JSON files found in directory: {input_dir}"
        )
    return paths


def _infer_feature_name(paths: List[str]) -> str:
    feature_names = set()
    for path in paths:
        name = os.path.basename(path)
        match = re.match(r"(.+)_layer\d+_evolution\.json$", name)
        if not match:
            raise ValueError(f"Unexpected per-layer filename format: {path}")
        feature_names.add(match.group(1))
    if len(feature_names) != 1:
        raise ValueError(
            f"Expected exactly one feature prefix in input directory, got: {sorted(feature_names)}"
        )
    return next(iter(feature_names))


def _load_layer_results(input_dir: str) -> List[Tuple[int, Dict[str, Any]]]:
    loaded = []
    for path in _collect_layer_json_paths(input_dir):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        layer_idx = data.get("layer_idx")
        if layer_idx is None:
            match = re.search(r"_layer(\d+)_evolution\.json$", os.path.basename(path))
            if not match:
                raise ValueError(f"Could not infer layer index from: {path}")
            layer_idx = int(match.group(1))
        loaded.append((int(layer_idx), data))
    loaded.sort(key=lambda item: item[0])
    return loaded


def _extract_layer_candidates(
    layer_entries: List[Tuple[int, Dict[str, Any]]],
) -> Dict[str, Any]:
    candidates: Dict[str, Any] = {
        "feature_file": None,
        "layers": [layer for layer, _ in layer_entries],
        "layer_candidates": {},
    }

    for layer, data in layer_entries:
        base_results = data.get("base_results", {})
        if candidates["feature_file"] is None:
            candidates["feature_file"] = base_results.get("feature_file")
        layer_results = base_results.get("layer_results", {})
        layer_result = layer_results.get(layer, layer_results.get(str(layer), {}))
        top_features = layer_result.get("top_features", [])
        if not top_features:
            candidates["layer_candidates"][str(layer)] = {"status": "missing_top_features"}
            continue

        base_vector, frc = top_features[0]
        candidates["layer_candidates"][str(layer)] = {
            "status": "ready",
            "layer_idx": int(layer),
            "base_vector": int(base_vector),
            "base_vectors": [int(curr_base_vector) for curr_base_vector, _ in top_features[:3]],
            "frc": float(frc),
        }

    return candidates


def _build_evolution_results(layer_entries: List[Tuple[int, Dict[str, Any]]]) -> Dict[str, Any]:
    feature_file = None
    evolution_data = {
        "layer_progression": {},
        "base_vector_evolution": defaultdict(dict),
    }

    for layer, data in layer_entries:
        base_results = data.get("base_results", {})
        if feature_file is None:
            feature_file = base_results.get("feature_file")
        layer_results = base_results.get("layer_results", {})
        layer_result = layer_results.get(layer, layer_results.get(str(layer), {}))
        top_features = layer_result.get("top_features", [])
        full_stats = layer_result.get("full_stats", {})
        if not top_features:
            continue

        evolution_data["layer_progression"][layer] = {
            "top_base_vectors": [int(bv) for bv, _ in top_features],
            "top_frc_scores": [float(score) for _, score in top_features],
            "total_base_vectors": len(full_stats),
        }
        for base_vec, _ in top_features:
            base_vec = int(base_vec)
            if base_vec in full_stats or str(base_vec) in full_stats:
                stats = full_stats.get(base_vec, full_stats.get(str(base_vec)))
                evolution_data["base_vector_evolution"][base_vec][layer] = stats

    return {
        "feature_file": feature_file,
        "layers_analyzed": [layer for layer, _ in layer_entries],
        "evolution_data": evolution_data,
    }


def _save_layer_candidates(
    layer_entries: List[Tuple[int, Dict[str, Any]]],
    output_path: str,
) -> None:
    candidates = _extract_layer_candidates(layer_entries)
    save_json_results(candidates, output_path)
    print(f"Saved layer candidates: {output_path}")


def _save_evolution_plot(
    layer_entries: List[Tuple[int, Dict[str, Any]]],
    output_path: str,
    metric: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required to generate plots."
        ) from exc

    evolution_results = _build_evolution_results(layer_entries)
    evolution_data = evolution_results["evolution_data"]

    plt.figure(figsize=(12, 8))
    for base_vec, layer_stats in evolution_data["base_vector_evolution"].items():
        if len(layer_stats) < 2:
            continue
        x_layers = sorted(layer_stats.keys())
        y_values = [layer_stats[layer][metric] for layer in x_layers]
        plt.plot(x_layers, y_values, marker="o", alpha=0.7, label=f"BV {base_vec}")

    plt.xlabel("Layer Index")
    plt.ylabel(f"{metric.upper()} Score")
    plt.title(f"Feature Evolution Across Layers - {metric.upper()}")
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved evolution plot: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Postprocess per-layer crosslayer outputs to generate plots or "
            "layer-candidate summaries."
        )
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing *_layer*_evolution.json files for one feature.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Default: same as --input-dir.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate a cross-layer plot from per-layer outputs.",
    )
    parser.add_argument(
        "--plot-metric",
        default="frc",
        help="Metric key for evolution plot (default: frc).",
    )
    parser.add_argument(
        "--export-layer-candidates",
        action="store_true",
        help="Export a layer-candidate summary JSON from per-layer outputs.",
    )

    args = parser.parse_args()
    if not args.plot and not args.export_layer_candidates:
        raise ValueError("Specify at least one of --plot or --export-layer-candidates.")

    output_dir = args.output_dir or args.input_dir
    os.makedirs(output_dir, exist_ok=True)

    layer_entries = _load_layer_results(args.input_dir)
    feature_name = _infer_feature_name(_collect_layer_json_paths(args.input_dir))

    if args.export_layer_candidates:
        candidates_path = os.path.join(output_dir, f"{feature_name}_layer_candidates.json")
        _save_layer_candidates(layer_entries, candidates_path)

    if args.plot:
        plot_path = os.path.join(output_dir, f"{feature_name}_{args.plot_metric}.png")
        _save_evolution_plot(layer_entries, plot_path, args.plot_metric)


if __name__ == "__main__":
    main()
