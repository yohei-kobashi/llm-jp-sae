"""
Cross-layer analysis for train.py SAE checkpoints.

This module mirrors LinguaLens's cross-layer analyzer but uses
TrainSaeLinguisticAnalyzer so it works with SAEs saved as:
    sae_layer{layer}.pth
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from sklearn.linear_model import ElasticNetCV, LassoCV

from analyzer_lingualens import TrainSaeLinguisticAnalyzer
from LinguaLens.lingualens.metrics import get_top_features_by_frc
from LinguaLens.lingualens.utils import (
    ProgressLogger,
    save_json_results,
    validate_layer_indices,
)

TOP_FEATURE_SAVE_COUNT = 100
LASSO_SELECTION_COUNTS = (10, 20, 50, 100)


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

    def analyze_feature_layer(
        self,
        feature_file: str,
        layer_idx: int,
        top_k: int = 10,
        layer_position: Optional[int] = None,
        total_layers: Optional[int] = None,
    ) -> Dict[str, Any]:
        layer_result = self.analyzer.analyze_feature_layer(
            feature_file=feature_file,
            layer_idx=layer_idx,
            top_k=top_k,
            layer_position=layer_position,
            total_layers=total_layers,
            retain_sentence_activations=True,
        )

        sentence_feature_rows = layer_result.pop("sentence_feature_rows", [])
        layer_results = layer_result.get("layer_results", {})
        single_layer = layer_results.get(layer_idx, layer_results.get(str(layer_idx), {}))
        full_stats = single_layer.get("full_stats", {})
        top_100_features = get_top_features_by_frc(full_stats, TOP_FEATURE_SAVE_COUNT)
        single_layer["top_100_features"] = top_100_features
        single_layer["top_100_base_vectors_desc"] = [
            int(base_vec) for base_vec, _ in top_100_features
        ]

        if top_100_features and sentence_feature_rows:
            stability_result = self._fit_intervention_feature_selector(
                sentence_feature_rows=sentence_feature_rows,
                top_features=top_100_features,
                full_stats=full_stats,
            )
            default_selection_key = "top_100"
            single_layer["intervention_features"] = stability_result["selected_base_vectors_by_count"][
                default_selection_key
            ]
            single_layer["intervention_feature_weights"] = stability_result["feature_weights"]
            single_layer["stability_selection"] = stability_result["metadata"]
            single_layer["lasso_selections"] = stability_result["selections_by_count"]
            single_layer["lasso_selected_base_vectors"] = stability_result[
                "selected_base_vectors_by_count"
            ]
            single_layer["elasticnet_feature_weights"] = stability_result[
                "elasticnet_feature_weights"
            ]
            single_layer["elasticnet_selection"] = stability_result["elasticnet_metadata"]
            single_layer["elasticnet_selections"] = stability_result[
                "elasticnet_selections_by_count"
            ]
            single_layer["elasticnet_selected_base_vectors"] = stability_result[
                "elasticnet_selected_base_vectors_by_count"
            ]

        return layer_result

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

    def _fit_intervention_feature_selector(
        self,
        sentence_feature_rows: List[Dict[str, Any]],
        top_features: List[List[Any]] | List[tuple[Any, Any]],
        full_stats: Dict[str, Any],
    ) -> Dict[str, Any]:
        def _standardize_matrix(matrix: np.ndarray) -> np.ndarray:
            feature_means = matrix.mean(axis=0, dtype=np.float64)
            feature_stds = matrix.std(axis=0, dtype=np.float64)
            feature_stds[feature_stds < 1e-6] = 1.0
            standardized = (matrix - feature_means) / feature_stds
            return standardized.astype(np.float64, copy=False)

        def _build_ranked_feature_weights(
            ranked_indices: List[int],
            coefficients: np.ndarray,
            scores: np.ndarray,
            score_name: str,
        ) -> List[Dict[str, Any]]:
            ranked_feature_weights: List[Dict[str, Any]] = []
            for rank_idx in ranked_indices:
                if rank_idx >= len(candidate_base_vectors):
                    continue
                base_vec = candidate_base_vectors[rank_idx]
                stats = full_stats.get(base_vec, full_stats.get(str(base_vec), {}))
                ranked_feature_weights.append(
                    {
                        "base_vector": int(base_vec),
                        "weight": float(coefficients[rank_idx]),
                        "abs_weight": float(abs(coefficients[rank_idx])),
                        score_name: float(scores[rank_idx]),
                        "frc": float(stats.get("frc", 0.0)),
                        "ps": float(stats.get("ps", 0.0)),
                        "pn": float(stats.get("pn", 0.0)),
                    }
                )
            return ranked_feature_weights

        def _build_topn_selections(
            ranked_indices: List[int],
            ranked_feature_weights: List[Dict[str, Any]],
        ) -> tuple[Dict[str, Any], Dict[str, List[int]]]:
            selections_by_count: Dict[str, Any] = {}
            selected_base_vectors_by_count: Dict[str, List[int]] = {}
            available_count = len(ranked_indices)
            for selection_count in LASSO_SELECTION_COUNTS:
                selection_key = f"top_{selection_count}"
                capped_count = min(selection_count, available_count)
                selected_rank_indices = ranked_indices[:capped_count]
                selected_base_vectors = [
                    candidate_base_vectors[idx] for idx in selected_rank_indices
                ]
                selected_base_vectors_by_count[selection_key] = selected_base_vectors
                selections_by_count[selection_key] = {
                    "selection_count_requested": int(selection_count),
                    "selection_count_returned": int(capped_count),
                    "selected_base_vectors": selected_base_vectors,
                    "feature_weights": ranked_feature_weights[:capped_count],
                }
            return selections_by_count, selected_base_vectors_by_count

        candidate_base_vectors = [int(base_vec) for base_vec, _ in top_features]
        labels = np.asarray(
            [float(row.get("label", 0)) for row in sentence_feature_rows],
            dtype=np.float32,
        )
        feature_matrix = np.zeros(
            (len(sentence_feature_rows), len(candidate_base_vectors)),
            dtype=np.float32,
        )

        for row_idx, row in enumerate(sentence_feature_rows):
            latent_activations = row.get("latent_activations", {})
            for col_idx, base_vec in enumerate(candidate_base_vectors):
                feature_matrix[row_idx, col_idx] = float(
                    latent_activations.get(base_vec, latent_activations.get(str(base_vec), 0.0))
        )

        coefficients = np.zeros(len(candidate_base_vectors), dtype=np.float32)
        stability_scores = np.zeros(len(candidate_base_vectors), dtype=np.float32)
        elasticnet_coefficients = np.zeros(len(candidate_base_vectors), dtype=np.float32)
        metadata: Dict[str, Any] = {
            "status": "not_run",
            "method": "lasso_stability_selection",
            "candidate_base_vectors": candidate_base_vectors,
            "num_examples": int(len(sentence_feature_rows)),
            "label_mapping": {
                "1": "odd_lines_original_text",
                "0": "even_lines_minimal_pair",
            },
        }
        elasticnet_metadata: Dict[str, Any] = {
            "status": "not_run",
            "method": "elasticnet_cv",
            "candidate_base_vectors": candidate_base_vectors,
            "num_examples": int(len(sentence_feature_rows)),
            "label_mapping": {
                "1": "odd_lines_original_text",
                "0": "even_lines_minimal_pair",
            },
        }

        can_fit = (
            len(candidate_base_vectors) > 0
            and len(sentence_feature_rows) >= 2
            and len(np.unique(labels)) >= 2
        )
        if can_fit:
            num_examples = len(sentence_feature_rows)
            subsample_fraction = 0.75
            subsample_size = max(2, int(np.ceil(num_examples * subsample_fraction)))
            num_bootstraps = 100
            selection_threshold = 0.6
            rng = np.random.default_rng(42)
            successful_fits = 0
            non_zero_counts = np.zeros(len(candidate_base_vectors), dtype=np.int32)

            for bootstrap_idx in range(num_bootstraps):
                sample_indices = rng.choice(
                    num_examples,
                    size=subsample_size,
                    replace=False,
                )
                sample_labels = labels[sample_indices]
                if len(np.unique(sample_labels)) < 2:
                    continue
                sample_matrix = _standardize_matrix(feature_matrix[sample_indices])
                cv_folds = max(2, min(5, len(sample_indices)))
                model = LassoCV(
                    cv=cv_folds,
                    random_state=42 + bootstrap_idx,
                    max_iter=10000,
                    precompute=False,
                )
                model.fit(sample_matrix, sample_labels)
                sample_coefficients = model.coef_.astype(np.float32, copy=False)
                non_zero_counts += (np.abs(sample_coefficients) > 1e-8).astype(np.int32)
                successful_fits += 1

            if successful_fits > 0:
                stability_scores = non_zero_counts.astype(np.float32) / float(successful_fits)
                standardized_feature_matrix = _standardize_matrix(feature_matrix)
                final_cv_folds = max(2, min(5, len(sentence_feature_rows)))
                final_model = LassoCV(
                    cv=final_cv_folds,
                    random_state=42,
                    max_iter=10000,
                    precompute=False,
                )
                final_model.fit(standardized_feature_matrix, labels)
                coefficients = final_model.coef_.astype(np.float32, copy=False)
                metadata.update(
                    {
                        "status": "fit",
                        "alpha": float(final_model.alpha_),
                        "intercept": float(final_model.intercept_),
                        "non_zero_feature_count": int(
                            np.count_nonzero(np.abs(coefficients) > 1e-8)
                        ),
                        "score_r2": float(final_model.score(standardized_feature_matrix, labels)),
                        "num_bootstraps": int(num_bootstraps),
                        "successful_bootstraps": int(successful_fits),
                        "subsample_fraction": float(subsample_fraction),
                        "selection_threshold": float(selection_threshold),
                    }
                )

                try:
                    elasticnet_model = ElasticNetCV(
                        l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99],
                        cv=final_cv_folds,
                        random_state=42,
                        max_iter=10000,
                    )
                    elasticnet_model.fit(standardized_feature_matrix, labels)
                    elasticnet_coefficients = elasticnet_model.coef_.astype(
                        np.float32, copy=False
                    )
                    elasticnet_metadata.update(
                        {
                            "status": "fit",
                            "alpha": float(elasticnet_model.alpha_),
                            "l1_ratio": float(elasticnet_model.l1_ratio_),
                            "intercept": float(elasticnet_model.intercept_),
                            "non_zero_feature_count": int(
                                np.count_nonzero(np.abs(elasticnet_coefficients) > 1e-8)
                            ),
                            "score_r2": float(
                                elasticnet_model.score(standardized_feature_matrix, labels)
                            ),
                        }
                    )
                except Exception as exc:
                    elasticnet_metadata.update(
                        {
                            "status": "fallback",
                            "reason": f"elasticnet_fit_failed: {exc}",
                        }
                    )
            else:
                metadata.update(
                    {
                        "status": "fallback",
                        "reason": "bootstrap_subsamples_single_label_only",
                        "num_bootstraps": int(num_bootstraps),
                        "successful_bootstraps": int(successful_fits),
                        "subsample_fraction": float(subsample_fraction),
                        "selection_threshold": float(selection_threshold),
                    }
                )
                elasticnet_metadata.update(
                    {
                        "status": "fallback",
                        "reason": "bootstrap_subsamples_single_label_only",
                    }
                )
        else:
            metadata["status"] = "fallback"
            elasticnet_metadata["status"] = "fallback"
            if len(candidate_base_vectors) == 0:
                metadata["reason"] = "no_candidate_base_vectors"
                elasticnet_metadata["reason"] = "no_candidate_base_vectors"
            elif len(sentence_feature_rows) < 2:
                metadata["reason"] = "not_enough_examples"
                elasticnet_metadata["reason"] = "not_enough_examples"
            else:
                metadata["reason"] = "single_label_only"
                elasticnet_metadata["reason"] = "single_label_only"

        ranked_indices = sorted(
            range(len(candidate_base_vectors)),
            key=lambda idx: (
                float(stability_scores[idx]),
                float(abs(coefficients[idx])),
            ),
            reverse=True,
        )
        feature_weights = _build_ranked_feature_weights(
            ranked_indices=ranked_indices,
            coefficients=coefficients,
            scores=stability_scores,
            score_name="stability_score",
        )
        selections_by_count, selected_base_vectors_by_count = _build_topn_selections(
            ranked_indices=ranked_indices,
            ranked_feature_weights=feature_weights,
        )

        elasticnet_ranked_indices = sorted(
            range(len(candidate_base_vectors)),
            key=lambda idx: float(abs(elasticnet_coefficients[idx])),
            reverse=True,
        )
        elasticnet_feature_weights = _build_ranked_feature_weights(
            ranked_indices=elasticnet_ranked_indices,
            coefficients=elasticnet_coefficients,
            scores=np.abs(elasticnet_coefficients),
            score_name="elasticnet_score",
        )
        (
            elasticnet_selections_by_count,
            elasticnet_selected_base_vectors_by_count,
        ) = _build_topn_selections(
            ranked_indices=elasticnet_ranked_indices,
            ranked_feature_weights=elasticnet_feature_weights,
        )

        metadata["selection_counts"] = [int(count) for count in LASSO_SELECTION_COUNTS]
        elasticnet_metadata["selection_counts"] = [
            int(count) for count in LASSO_SELECTION_COUNTS
        ]

        return {
            "selected_base_vectors": selected_base_vectors_by_count.get("top_100", []),
            "selected_base_vectors_by_count": selected_base_vectors_by_count,
            "feature_weights": feature_weights,
            "metadata": metadata,
            "selections_by_count": selections_by_count,
            "elasticnet_feature_weights": elasticnet_feature_weights,
            "elasticnet_metadata": elasticnet_metadata,
            "elasticnet_selections_by_count": elasticnet_selections_by_count,
            "elasticnet_selected_base_vectors_by_count": (
                elasticnet_selected_base_vectors_by_count
            ),
        }


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


def _build_single_layer_result(
    base_results: Dict[str, Any],
    layer: int,
) -> Dict[str, Any]:
    layer_results = base_results.get("layer_results", {})
    layer_result = layer_results.get(layer, layer_results.get(str(layer), {}))

    return {
        "feature_file": base_results.get("feature_file"),
        "layer_idx": layer,
        "base_results": {
            "feature_file": base_results.get("feature_file"),
            "total_examples": base_results.get("total_examples"),
            "layers_analyzed": [layer],
            "top_k": base_results.get("top_k"),
            "layer_results": {layer: layer_result} if layer_result else {},
            "unified_results": [
                item
                for item in base_results.get("unified_results", [])
                if int(item.get("layer", -1)) == int(layer)
            ],
        },
    }


def _save_split_layer_results(
    base_results: Dict[str, Any],
    layers: List[int],
    output_dir: str,
    feature_name: str,
) -> List[str]:
    output_paths = []
    for layer in layers:
        single_layer_result = _build_single_layer_result(base_results, layer)
        output_path = os.path.join(output_dir, f"{feature_name}_layer{layer}_evolution.json")
        save_json_results(single_layer_result, output_path)
        output_paths.append(output_path)
        print(f"Saved layer result: {output_path}")
    return output_paths


def _is_completed_layer_output(
    output_path: str,
    expected_layer: int,
) -> bool:
    if not os.path.exists(output_path):
        return False
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return False

    layer_idx = data.get("layer_idx")
    if layer_idx is not None and int(layer_idx) != int(expected_layer):
        return False

    base_results = data.get("base_results", {})
    layer_results = base_results.get("layer_results", {})
    layer_result = layer_results.get(expected_layer, layer_results.get(str(expected_layer), {}))
    return bool(layer_result) and "top_features" in layer_result


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
        "--resume",
        action="store_true",
        help="Skip already-saved per-layer outputs in single-feature mode.",
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
            feature_name = os.path.splitext(os.path.basename(args.feature_file))[0]
            print("[4/4] Saving outputs")
            completed_layers = 0
            for layer_position, layer in enumerate(layers, start=1):
                output_path = os.path.join(
                    args.output_dir, f"{feature_name}_layer{layer}_evolution.json"
                )
                if args.resume and _is_completed_layer_output(output_path, layer):
                    print(f"[layer {layer_position}/{len(layers)}] skipping saved output: {output_path}")
                    completed_layers += 1
                    continue

                layer_result = analyzer.analyze_feature_layer(
                    feature_file=args.feature_file,
                    layer_idx=layer,
                    top_k=args.top_k,
                    layer_position=layer_position,
                    total_layers=len(layers),
                )
                single_layer_result = _build_single_layer_result(layer_result, layer)
                save_json_results(single_layer_result, output_path)
                print(f"Saved layer result: {output_path}")
                completed_layers += 1

            print(
                f"Single-feature analysis complete. Saved or reused {completed_layers}/{len(layers)} layers."
            )

            if args.plot:
                print(
                    "Skipping plot generation in single-feature mode. "
                    "Use postprocess_crosslayer_lingualens.py on the per-layer outputs."
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
