"""
Cross-layer analysis for train.py SAE checkpoints.

This module mirrors LinguaLens's cross-layer analyzer but uses
TrainSaeLinguisticAnalyzer so it works with SAEs saved as:
    sae_layer{layer}.pth
"""

from __future__ import annotations

import argparse
import os
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

import torch

from analyzer_lingualens import TrainSaeLinguisticAnalyzer
from LinguaLens.lingualens.utils import (
    ProgressLogger,
    save_json_results,
    validate_layer_indices,
)


class TrainSaeCrossLayerAnalyzer:
    """
    Cross-layer analyzer compatible with train.py SAE checkpoints.

    This class mirrors LinguaLens.lingualens.crosslayer.CrossLayerAnalyzer
    and delegates per-layer analysis to TrainSaeLinguisticAnalyzer.
    """

    def __init__(
        self,
        model_path: str,
        sae_path_template: str = "/path/to/sae/sae_layer{}.pth",
        device: Optional[str] = None,
        k: int = 32,
        normalization: str = "Scalar",
        batch_size: int = 8,
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        self.analyzer = TrainSaeLinguisticAnalyzer(
            model_path=model_path,
            sae_path_template=sae_path_template,
            device=device,
            k=k,
            normalization=normalization,
            batch_size=batch_size,
            torch_dtype=torch_dtype,
        )
        self.model_path = model_path
        self.sae_path_template = sae_path_template

    def analyze_feature_evolution(
        self,
        feature_file: str,
        layers: List[int],
        top_k: int = 10,
    ) -> Dict[str, Any]:
        start_time = time.perf_counter()
        print(f"[crosslayer] feature file: {feature_file}")
        print(f"[crosslayer] analyzing {len(layers)} layers: {','.join(map(str, layers))}")
        print("[crosslayer] phase 1/3: per-layer analysis")

        base_results = self.analyzer.analyze_feature(feature_file, layers, top_k)
        print("[crosslayer] phase 2/3: extracting cross-layer evolution")
        evolution_data = self._extract_evolution_data(base_results, layers)
        print("[crosslayer] phase 3/3: computing cross-layer statistics")
        cross_layer_stats = self._compute_cross_layer_stats(evolution_data)

        completed_layers = len(
            [
                layer
                for layer, result in base_results.get("layer_results", {}).items()
                if "top_features" in result
            ]
        )
        print(
            f"[crosslayer] done: completed_layers={completed_layers}/{len(layers)}, "
            f"elapsed={time.perf_counter() - start_time:.1f}s"
        )

        return {
            "feature_file": feature_file,
            "layers_analyzed": layers,
            "evolution_data": evolution_data,
            "cross_layer_stats": cross_layer_stats,
            "base_results": base_results,
        }

    def _extract_evolution_data(
        self,
        base_results: Dict[str, Any],
        layers: List[int],
    ) -> Dict[str, Any]:
        evolution_data = {
            "layer_progression": {},
            "base_vector_evolution": defaultdict(dict),
            "frc_evolution": {},
            "stability_analysis": {},
        }

        for layer in layers:
            layer_result = base_results.get("layer_results", {}).get(layer, {})
            if "top_features" not in layer_result:
                continue

            top_features = layer_result["top_features"]
            full_stats = layer_result["full_stats"]

            evolution_data["layer_progression"][layer] = {
                "top_base_vectors": [bv for bv, _ in top_features],
                "top_frc_scores": [score for _, score in top_features],
                "total_base_vectors": len(full_stats),
            }

            for base_vec, _ in top_features:
                if base_vec in full_stats:
                    evolution_data["base_vector_evolution"][base_vec][layer] = full_stats[
                        base_vec
                    ]

        return evolution_data

    def _compute_cross_layer_stats(
        self, evolution_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        stats = {
            "layer_overlap": {},
            "persistence_analysis": {},
            "emergence_analysis": {},
            "peak_layers": {},
        }

        layers = sorted(evolution_data["layer_progression"].keys())

        for i in range(len(layers) - 1):
            layer1, layer2 = layers[i], layers[i + 1]
            bv1 = set(evolution_data["layer_progression"][layer1]["top_base_vectors"])
            bv2 = set(evolution_data["layer_progression"][layer2]["top_base_vectors"])

            overlap = len(bv1 & bv2)
            union = len(bv1 | bv2)
            stats["layer_overlap"][f"{layer1}->{layer2}"] = {
                "overlap_count": overlap,
                "jaccard_similarity": overlap / union if union > 0 else 0.0,
                "persistence_rate": overlap / len(bv1) if len(bv1) > 0 else 0.0,
            }

        for base_vec, layer_stats in evolution_data["base_vector_evolution"].items():
            layers_present = sorted(layer_stats.keys())
            frc_scores = [layer_stats[l]["frc"] for l in layers_present]

            stats["persistence_analysis"][base_vec] = {
                "layers_present": layers_present,
                "persistence_length": len(layers_present),
                "frc_scores": frc_scores,
                "max_frc": max(frc_scores) if frc_scores else 0.0,
                "avg_frc": sum(frc_scores) / len(frc_scores) if frc_scores else 0.0,
                "peak_layer": (
                    layers_present[frc_scores.index(max(frc_scores))]
                    if frc_scores
                    else None
                ),
            }

        return stats

    def compare_features_across_layers(
        self,
        feature_files: List[str],
        layers: List[int],
        output_dir: str,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        os.makedirs(output_dir, exist_ok=True)

        comparison_results = {
            "features": [os.path.splitext(os.path.basename(f))[0] for f in feature_files],
            "layers": layers,
            "feature_results": {},
            "cross_feature_analysis": {},
        }

        progress = ProgressLogger(len(feature_files), "Cross-layer feature comparison")

        for feature_file in feature_files:
            feature_name = os.path.splitext(os.path.basename(feature_file))[0]
            try:
                evolution_results = self.analyze_feature_evolution(
                    feature_file, layers, top_k
                )
                comparison_results["feature_results"][feature_name] = evolution_results

                feature_output = os.path.join(output_dir, f"{feature_name}_evolution.json")
                save_json_results(evolution_results, feature_output)
            except Exception as exc:
                print(f"Failed to analyze {feature_name}: {exc}")
            finally:
                progress.update()

        progress.finish()

        comparison_results["cross_feature_analysis"] = self._compute_cross_feature_stats(
            comparison_results["feature_results"], layers
        )

        summary_file = os.path.join(output_dir, "cross_layer_comparison.json")
        save_json_results(comparison_results, summary_file)

        return comparison_results

    def _compute_cross_feature_stats(
        self,
        feature_results: Dict[str, Dict[str, Any]],
        layers: List[int],
    ) -> Dict[str, Any]:
        cross_stats = {
            "layer_diversity": {},
            "feature_similarity": {},
            "layer_specialization": {},
        }

        for layer in layers:
            layer_base_vectors = set()
            feature_count = 0

            for results in feature_results.values():
                layer_progression = results.get("evolution_data", {}).get(
                    "layer_progression", {}
                )
                if layer in layer_progression:
                    bvs = layer_progression[layer]["top_base_vectors"]
                    layer_base_vectors.update(bvs)
                    feature_count += 1

            cross_stats["layer_diversity"][layer] = {
                "unique_base_vectors": len(layer_base_vectors),
                "features_present": feature_count,
                "avg_bv_per_feature": (
                    len(layer_base_vectors) / feature_count if feature_count > 0 else 0
                ),
            }

        return cross_stats

    def generate_evolution_plot(
        self,
        evolution_results: Dict[str, Any],
        output_path: str,
        metric: str = "frc",
    ) -> None:
        try:
            import matplotlib.pyplot as plt
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "matplotlib is required only when --plot is used."
            ) from exc

        plt.figure(figsize=(12, 8))

        evolution_data = evolution_results["evolution_data"]
        for base_vec, layer_stats in evolution_data["base_vector_evolution"].items():
            if len(layer_stats) < 3:
                continue
            x_layers = sorted(layer_stats.keys())
            y_values = [layer_stats[l][metric] for l in x_layers]
            plt.plot(x_layers, y_values, marker="o", alpha=0.7, label=f"BV {base_vec}")

        plt.xlabel("Layer Index")
        plt.ylabel(f"{metric.upper()} Score")
        plt.title(f"Feature Evolution Across Layers - {metric.upper()}")
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()

        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Evolution plot saved to {output_path}")

    def clear_cache(self):
        self.analyzer.clear_cache()


# Backward-compatible alias if users want the same class name as LinguaLens.
CrossLayerAnalyzer = TrainSaeCrossLayerAnalyzer


def _parse_layers(layers_arg: str) -> List[int]:
    layers = [int(x.strip()) for x in layers_arg.split(",") if x.strip()]
    return validate_layer_indices(layers)


def _parse_torch_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")
    return mapping[dtype_name]


def _extract_layer_candidates(results: Dict[str, Any], layers: List[int]) -> Dict[str, Any]:
    layer_results = results.get("base_results", {}).get("layer_results", {})
    candidates: Dict[str, Any] = {
        "feature_file": results.get("feature_file"),
        "layers": layers,
        "layer_candidates": {},
    }

    for layer in layers:
        result = layer_results.get(layer, layer_results.get(str(layer), {}))
        top_features = result.get("top_features", [])
        if not top_features:
            candidates["layer_candidates"][str(layer)] = {
                "status": "missing_top_features",
            }
            continue

        base_vector, frc = top_features[0]
        candidates["layer_candidates"][str(layer)] = {
            "status": "ready",
            "layer_idx": int(layer),
            "base_vector": int(base_vector),
            "frc": float(frc),
        }

    return candidates


def _save_layer_candidates(
    results: Dict[str, Any],
    layers: List[int],
    output_path: str,
) -> None:
    candidates = _extract_layer_candidates(results, layers)
    save_json_results(candidates, output_path)
    print(f"Saved layer candidates: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-layer analysis for train.py SAE checkpoints."
    )
    parser.add_argument(
        "--model-path",
        required=True,
        default="llm-jp/llm-jp-3-1.8b",
        help="Path or HF name of the base LLM."
    )
    parser.add_argument(
        "--sae-path-template",
        required=True,
        default="sae/n_d_16/k_32/nl_Scalar/ckpt_0988240/lr_0.001/sae_layer{}.pth",
        help="SAE checkpoint template, e.g. /path/to/sae_layer{}.pth",
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
    parser.add_argument("--k", type=int, default=32, help="SAE top-k activation setting.")
    parser.add_argument("--normalization", default="Scalar", help="Activation normalization mode.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for analysis.")
    parser.add_argument(
        "--torch-dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
        help="Torch dtype for loading the base model.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device override (e.g. cuda:0 or cpu). Default: auto.",
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
        "--export-layer-candidates",
        action="store_true",
        help="If set in single feature mode, save each layer's top base vector as JSON.",
    )
    parser.add_argument(
        "--layer-candidates-output",
        default=None,
        help="Optional path for layer-candidate JSON. Default: <output-dir>/<feature>_layer_candidates.json",
    )

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
    analyzer = TrainSaeCrossLayerAnalyzer(
        model_path=args.model_path,
        sae_path_template=args.sae_path_template,
        device=args.device,
        k=args.k,
        normalization=args.normalization,
        batch_size=args.batch_size,
        torch_dtype=_parse_torch_dtype(args.torch_dtype),
    )

    try:
        if args.feature_file:
            print(f"[3/4] Running single-feature analysis: {args.feature_file}")
            results = analyzer.analyze_feature_evolution(
                feature_file=args.feature_file,
                layers=layers,
                top_k=args.top_k,
            )
            feature_name = os.path.splitext(os.path.basename(args.feature_file))[0]
            output_json = os.path.join(args.output_dir, f"{feature_name}_evolution.json")
            print("[4/4] Saving outputs")
            save_json_results(results, output_json)
            print(f"Saved evolution result: {output_json}")

            if args.export_layer_candidates:
                candidates_output = args.layer_candidates_output or os.path.join(
                    args.output_dir, f"{feature_name}_layer_candidates.json"
                )
                _save_layer_candidates(results, layers, candidates_output)

            if args.plot:
                plot_path = os.path.join(
                    args.output_dir, f"{feature_name}_{args.plot_metric}.png"
                )
                analyzer.generate_evolution_plot(
                    results, plot_path, metric=args.plot_metric
                )

        if args.feature_files:
            print(
                f"[3/4] Running multi-feature analysis: {len(args.feature_files)} files"
            )
            compare_results = analyzer.compare_features_across_layers(
                feature_files=args.feature_files,
                layers=layers,
                output_dir=args.output_dir,
                top_k=args.top_k,
            )
            print("[4/4] Saving outputs")
            print(
                "Feature comparison complete. "
                f"Analyzed: {len(compare_results.get('feature_results', {}))}"
            )
            if args.plot:
                feature_results = compare_results.get("feature_results", {})
                for feature_name, evolution_results in feature_results.items():
                    plot_path = os.path.join(
                        args.output_dir, f"{feature_name}_{args.plot_metric}.png"
                    )
                    try:
                        analyzer.generate_evolution_plot(
                            evolution_results,
                            plot_path,
                            metric=args.plot_metric,
                        )
                    except Exception as exc:
                        print(f"Failed to plot {feature_name}: {exc}")
        print(f"Total elapsed: {time.perf_counter() - total_start:.1f}s")
    finally:
        analyzer.clear_cache()


if __name__ == "__main__":
    main()
