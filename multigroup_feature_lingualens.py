"""
Multigroup feature analysis module for cross-group feature comparison.

This module supports comparing 3+ language/language-feature groups
using layer-wise base-vector overlaps.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Set

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from LinguaLens.lingualens.utils import ProgressLogger, save_json_results


class MultiGroupFeatureAnalyzer:
    """Analyzer for multi-group feature comparison (3+ groups)."""

    def __init__(self, vector_data_dir: Optional[str] = None):
        self.vector_data_dir = vector_data_dir or "data/vectors"

    def load_vector_data(self, filepath: str) -> List[Set[str]]:
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Vector file not found: {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
        return [set(line.split()) for line in lines]

    @staticmethod
    def _jaccard(a: Set[str], b: Set[str]) -> float:
        union = len(a | b)
        if union == 0:
            return 0.0
        return len(a & b) / union

    def _load_group_vectors(self, groups: List[Dict[str, Any]]) -> Dict[str, List[List[Set[str]]]]:
        loaded: Dict[str, List[List[Set[str]]]] = {}
        for group in groups:
            name = group["name"]
            features = group["features"]
            if not isinstance(features, list) or not features:
                raise ValueError(f"Group '{name}' must have a non-empty features list.")

            feature_vecs = []
            for feat in features:
                path = os.path.join(self.vector_data_dir, f"{feat}.txt")
                feature_vecs.append(self.load_vector_data(path))
            loaded[name] = feature_vecs
        return loaded

    def analyze_multigroup_feature_similarity(
        self,
        groups: List[Dict[str, Any]],
        output_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Analyze layer-wise similarity for 3+ groups.

        Args:
            groups: [{"name": str, "features": [str, ...]}, ...]
            output_path: optional JSON output path
        """
        if len(groups) < 3:
            raise ValueError("Use at least 3 groups for multigroup analysis.")

        group_names = [g["name"] for g in groups]
        loaded = self._load_group_vectors(groups)

        max_layers = 0
        for feature_vecs in loaded.values():
            for vecs in feature_vecs:
                max_layers = max(max_layers, len(vecs))

        results: Dict[str, Any] = {
            "num_groups": len(groups),
            "group_names": group_names,
            "groups": groups,
            "num_layers": max_layers,
            "layer_results": [],
            "overall_statistics": {},
        }

        progress = ProgressLogger(max_layers, "Analyzing multigroup layers")

        for layer_idx in range(max_layers):
            group_sets: Dict[str, Set[str]] = {}
            for name, feature_vecs in loaded.items():
                union_set: Set[str] = set()
                for vecs in feature_vecs:
                    if layer_idx < len(vecs):
                        union_set |= vecs[layer_idx]
                group_sets[name] = union_set

            n = len(group_names)
            pairwise = np.zeros((n, n), dtype=float)
            for i, ni in enumerate(group_names):
                for j, nj in enumerate(group_names):
                    pairwise[i, j] = self._jaccard(group_sets[ni], group_sets[nj])

            all_sets = [group_sets[name] for name in group_names]
            inter_all = set.intersection(*all_sets) if all_sets else set()
            union_all = set.union(*all_sets) if all_sets else set()
            global_intersection_ratio = len(inter_all) / len(union_all) if union_all else 0.0

            off_diag = pairwise[~np.eye(n, dtype=bool)]
            mean_pairwise_jaccard = float(off_diag.mean()) if off_diag.size else 0.0

            coverage = {}
            for name in group_names:
                this_set = group_sets[name]
                others_union = set().union(*[group_sets[k] for k in group_names if k != name])
                coverage[name] = len(this_set & others_union) / len(this_set) if this_set else 0.0

            results["layer_results"].append(
                {
                    "layer": layer_idx,
                    "group_vector_counts": {k: len(v) for k, v in group_sets.items()},
                    "pairwise_jaccard": pairwise.tolist(),
                    "global_intersection_ratio": global_intersection_ratio,
                    "mean_pairwise_jaccard": mean_pairwise_jaccard,
                    "coverage_per_group": coverage,
                }
            )
            progress.update()

        progress.finish()

        g_ir = [x["global_intersection_ratio"] for x in results["layer_results"]]
        mpj = [x["mean_pairwise_jaccard"] for x in results["layer_results"]]

        results["overall_statistics"] = {
            "mean_global_intersection_ratio": float(np.mean(g_ir)) if g_ir else 0.0,
            "std_global_intersection_ratio": float(np.std(g_ir)) if g_ir else 0.0,
            "mean_pairwise_jaccard": float(np.mean(mpj)) if mpj else 0.0,
            "std_pairwise_jaccard": float(np.std(mpj)) if mpj else 0.0,
            "best_layer_by_global_intersection": int(np.argmax(g_ir)) if g_ir else None,
            "best_layer_by_mean_pairwise": int(np.argmax(mpj)) if mpj else None,
        }

        if output_path:
            save_json_results(results, output_path)
            print(f"Multigroup analysis complete. Results saved to {output_path}")

        return results

    def generate_layer_pairwise_heatmaps(
        self,
        results: Dict[str, Any],
        output_dir: str,
        figsize: tuple = (8, 6),
    ) -> None:
        os.makedirs(output_dir, exist_ok=True)
        group_names = results["group_names"]

        for layer_result in results["layer_results"]:
            layer = layer_result["layer"]
            matrix = np.array(layer_result["pairwise_jaccard"])

            plt.figure(figsize=figsize)
            sns.heatmap(
                matrix,
                vmin=0.0,
                vmax=1.0,
                cmap="viridis",
                xticklabels=group_names,
                yticklabels=group_names,
                cbar_kws={"label": "Jaccard Similarity"},
                annot=False,
            )
            plt.title(f"Pairwise Group Similarity (Layer {layer:02d})")
            plt.xlabel("Group")
            plt.ylabel("Group")
            plt.tight_layout()

            out = os.path.join(output_dir, f"layer_{layer:02d}_pairwise_heatmap.png")
            plt.savefig(out, dpi=300, bbox_inches="tight")
            plt.close()

    def generate_global_summary_plot(
        self,
        results: Dict[str, Any],
        output_path: str,
        figsize: tuple = (10, 5),
    ) -> None:
        layers = [x["layer"] for x in results["layer_results"]]
        g_ir = [x["global_intersection_ratio"] for x in results["layer_results"]]
        mpj = [x["mean_pairwise_jaccard"] for x in results["layer_results"]]

        plt.figure(figsize=figsize)
        plt.plot(layers, g_ir, marker="o", label="Global Intersection / Union")
        plt.plot(layers, mpj, marker="s", label="Mean Pairwise Jaccard")
        plt.xlabel("Layer")
        plt.ylabel("Score")
        plt.title("Multigroup Similarity Across Layers")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()

        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

    def export_similarity_report(self, results: Dict[str, Any], output_path: str) -> None:
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("# Multigroup Feature Similarity Report\n\n")
            f.write(f"- Number of groups: {results['num_groups']}\n")
            f.write(f"- Group names: {', '.join(results['group_names'])}\n")
            f.write(f"- Number of layers: {results['num_layers']}\n\n")

            overall = results.get("overall_statistics", {})
            f.write("## Overall Statistics\n")
            f.write(
                f"- Mean Global Intersection Ratio: {overall.get('mean_global_intersection_ratio', 0.0):.4f}\n"
            )
            f.write(
                f"- Mean Pairwise Jaccard: {overall.get('mean_pairwise_jaccard', 0.0):.4f}\n"
            )
            f.write(
                f"- Best Layer (Global): {overall.get('best_layer_by_global_intersection')}\n"
            )
            f.write(
                f"- Best Layer (Pairwise): {overall.get('best_layer_by_mean_pairwise')}\n\n"
            )

            f.write("## Per-Layer Summary\n")
            for lr in results.get("layer_results", []):
                f.write(f"### Layer {lr['layer']:02d}\n")
                f.write(f"- Global Intersection Ratio: {lr['global_intersection_ratio']:.4f}\n")
                f.write(f"- Mean Pairwise Jaccard: {lr['mean_pairwise_jaccard']:.4f}\n")
                cov_str = ", ".join(
                    [f"{k}: {v:.3f}" for k, v in lr.get("coverage_per_group", {}).items()]
                )
                f.write(f"- Coverage per Group: {cov_str}\n\n")

        print(f"Detailed report saved to {output_path}")
