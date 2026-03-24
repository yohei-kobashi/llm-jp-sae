"""
Cross-layer analysis for Gemma Scope sparse dictionaries.

This module reuses the cross-layer aggregation logic from
crosslayer_lingualens.py and swaps in GemmaScopeLinguisticAnalyzer for
per-layer analysis.
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Optional

from crosslayer_lingualens import (
    TrainSaeCrossLayerAnalyzer,
    _build_single_layer_result,
    _is_completed_layer_output,
    _parse_layers,
    _parse_torch_dtype,
)
from gemmascope_analyzer_lingualens import GemmaScopeLinguisticAnalyzer
from LinguaLens.lingualens.metrics import get_top_features_by_frc
from LinguaLens.lingualens.utils import save_json_results

TOP_FEATURE_SAVE_COUNT = 100


class GemmaScopeCrossLayerAnalyzer(TrainSaeCrossLayerAnalyzer):
    def __init__(
        self,
        model_path: str,
        sae_path_template: str,
        device: Optional[str] = None,
        batch_size: int = 8,
        torch_dtype=None,
        prepend_bos: bool = True,
        fold_activation_scale: bool = False,
    ):
        self.analyzer = GemmaScopeLinguisticAnalyzer(
            model_path=model_path,
            sae_path_template=sae_path_template,
            device=device,
            batch_size=batch_size,
            torch_dtype=torch_dtype,
            prepend_bos=prepend_bos,
            fold_activation_scale=fold_activation_scale,
        )
        self.model_path = model_path
        self.sae_path_template = sae_path_template

    def analyze_feature_layers(
        self,
        feature_file: str,
        layers: list[int],
        top_k: int = 10,
    ) -> dict:
        base_results = self.analyzer.analyze_feature(feature_file, layers, top_k)
        layer_results = base_results.get("layer_results", {})

        for layer in layers:
            single_layer = layer_results.get(layer, layer_results.get(str(layer), {}))
            if not single_layer or "full_stats" not in single_layer:
                continue
            full_stats = single_layer.get("full_stats", {})
            top_100_features = get_top_features_by_frc(full_stats, TOP_FEATURE_SAVE_COUNT)
            single_layer["top_100_features"] = top_100_features
            single_layer["top_100_base_vectors_desc"] = [
                int(base_vec) for base_vec, _ in top_100_features
            ]
            single_layer["stability_selection"] = {
                "status": "not_run",
                "reason": "model_selection_removed",
                "candidate_base_vectors": single_layer["top_100_base_vectors_desc"],
            }

        return base_results


CrossLayerAnalyzer = GemmaScopeCrossLayerAnalyzer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-layer analysis for Gemma Scope sparse dictionaries."
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Base model name for TransformerLens HookedTransformer.from_pretrained().",
    )
    parser.add_argument(
        "--sae-path-template",
        required=True,
        help="SAELens-compatible template, e.g. gemma-scope-...:layer_{}_width_16k_l0_small",
    )
    parser.add_argument(
        "--layers",
        required=True,
        help="Comma-separated layer indices, e.g. 0,7,15,19,25,31",
    )
    parser.add_argument(
        "--feature-file",
        default=None,
        help="Single feature text file for analyze_feature_evolution.",
    )
    parser.add_argument(
        "--feature-files",
        nargs="+",
        default=None,
        help="Multiple feature text files for compare_features_across_layers.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save output JSON/plots.",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Top K base vectors per layer.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for analysis.")
    parser.add_argument(
        "--torch-dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
        help="Torch dtype placeholder kept for CLI compatibility.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device override (e.g. cuda:0 or cpu). Default: auto.",
    )
    parser.add_argument(
        "--prepend-bos",
        dest="prepend_bos",
        action="store_true",
        help="Prepend BOS before running the hooked model.",
    )
    parser.add_argument(
        "--no-prepend-bos",
        dest="prepend_bos",
        action="store_false",
        help="Disable BOS prepending before running the hooked model.",
    )
    parser.add_argument(
        "--fold-activation-scale",
        action="store_true",
        help="Fold dataset activation scale into SparseDictionary weights.",
    )
    parser.add_argument(
        "--plot-metric",
        default="frc",
        help="Metric key for evolution plot (default: frc).",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="If set, save an evolution plot for single feature mode.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip already-saved per-layer outputs in single-feature mode.",
    )
    parser.set_defaults(prepend_bos=True)
    args = parser.parse_args()
    if not args.feature_file and not args.feature_files:
        raise ValueError("Specify --feature-file or --feature-files.")

    os.makedirs(args.output_dir, exist_ok=True)
    layers = _parse_layers(args.layers)
    total_start = time.perf_counter()

    print("[1/4] Preparing cross-layer analysis")
    print(f"      model: {args.model_path}")
    print(f"      layers: {','.join(map(str, layers))}")
    print(f"      output_dir: {args.output_dir}")

    print("[2/4] Initializing analyzer")
    analyzer = GemmaScopeCrossLayerAnalyzer(
        model_path=args.model_path,
        sae_path_template=args.sae_path_template,
        device=args.device,
        batch_size=args.batch_size,
        torch_dtype=_parse_torch_dtype(args.torch_dtype),
        prepend_bos=args.prepend_bos,
        fold_activation_scale=args.fold_activation_scale,
    )

    try:
        if args.feature_file:
            print(f"[3/4] Running single-feature analysis: {args.feature_file}")
            feature_name = os.path.splitext(os.path.basename(args.feature_file))[0]
            print("[4/4] Saving outputs")
            completed_layers = 0
            for layer_position, layer in enumerate(layers, start=1):
                output_path = os.path.join(
                    args.output_dir, f"{feature_name}_layer{layer}_evolution.json"
                )
                if args.resume and _is_completed_layer_output(output_path, layer):
                    print(
                        f"[layer {layer_position}/{len(layers)}] skipping saved output: {output_path}"
                    )
                    completed_layers += 1
                    continue

                layer_result = analyzer.analyze_feature_layer(
                    feature_file=args.feature_file,
                    layer_idx=layer,
                    top_k=args.top_k,
                    layer_position=layer_position,
                    total_layers=len(layers),
                )
                save_json_results(_build_single_layer_result(layer_result, layer), output_path)
                print(f"[layer {layer_position}/{len(layers)}] saved: {output_path}")
                completed_layers += 1

            print(
                f"Single-feature analysis complete: completed_layers={completed_layers}/{len(layers)}"
            )
        else:
            print(f"[3/4] Running multi-feature comparison: {len(args.feature_files)} files")
            comparison_results = analyzer.compare_features_across_layers(
                feature_files=args.feature_files,
                layers=layers,
                output_dir=args.output_dir,
                top_k=args.top_k,
            )
            print(
                f"[4/4] Saved summary: {os.path.join(args.output_dir, 'cross_layer_comparison.json')}"
            )
            print(
                f"Compared {len(comparison_results.get('feature_results', {}))} features."
            )

        if args.feature_file and args.plot:
            full_result = analyzer.analyze_feature_evolution(
                feature_file=args.feature_file,
                layers=layers,
                top_k=args.top_k,
            )
            plot_path = os.path.join(
                args.output_dir,
                f"{os.path.splitext(os.path.basename(args.feature_file))[0]}_{args.plot_metric}.png",
            )
            analyzer.generate_evolution_plot(full_result, plot_path, metric=args.plot_metric)

        print(f"Elapsed: {time.perf_counter() - total_start:.1f}s")
    finally:
        analyzer.clear_cache()


if __name__ == "__main__":
    main()
