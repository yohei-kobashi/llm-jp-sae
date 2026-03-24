#!/usr/bin/env python3

from __future__ import annotations

import argparse
import itertools
import json
import os
import re
from collections import Counter, defaultdict
from glob import glob
from typing import Any, Dict, List, Optional, Tuple

from LinguaLens.lingualens.utils import save_json_results


BASE_VECTOR_SOURCE_CHOICES = (
    "top_features",
    "top_100_features",
    "top_100_base_vectors_desc",
    "intervention_features",
    "lasso",
    "elasticnet",
)
DEFAULT_TARGET_FEATURE_NAMES = (
    "will",
    "can",
    "could",
    "may",
    "might",
    "must",
    "should",
    "would",
    "suppose",
    "know",
)
FEATURE_ORDER_PRIORITY = {
    name: idx for idx, name in enumerate(DEFAULT_TARGET_FEATURE_NAMES)
}
DEFAULT_OVERLAP_SELECTIONS = {
    "top3": ("top_features", "top_3"),
    "top100": ("top_100_features", "top_100"),
    "lasso_top10": ("lasso", "top_10"),
    "lasso_top20": ("lasso", "top_20"),
    "lasso_top50": ("lasso", "top_50"),
    "lasso_top100": ("lasso", "top_100"),
    "elasticnet_top10": ("elasticnet", "top_10"),
    "elasticnet_top20": ("elasticnet", "top_20"),
    "elasticnet_top50": ("elasticnet", "top_50"),
    "elasticnet_top100": ("elasticnet", "top_100"),
}


def _collect_layer_json_paths(input_dir: str) -> List[str]:
    direct_paths = glob(os.path.join(input_dir, "*_layer*_evolution.json"))
    nested_paths = glob(
        os.path.join(input_dir, "*", "crosslayer", "*_layer*_evolution.json")
    )
    paths = sorted(set(direct_paths + nested_paths))
    if not paths:
        raise FileNotFoundError(
            "No per-layer crosslayer JSON files found in directory: "
            f"{input_dir}. Expected either '<input-dir>/*_layer*_evolution.json' "
            "or '<input-dir>/<feature>/crosslayer/*_layer*_evolution.json'."
        )
    return paths


def _parse_feature_name_from_path(path: str) -> str:
    normalized_path = os.path.normpath(path)
    path_parts = normalized_path.split(os.sep)
    if len(path_parts) >= 3 and path_parts[-2] == "crosslayer":
        return path_parts[-3]

    name = os.path.basename(path)
    for feature_name in sorted(DEFAULT_TARGET_FEATURE_NAMES, key=len, reverse=True):
        if name == f"{feature_name}.json" or name.startswith(f"{feature_name}_"):
            return feature_name

    match = re.match(r"(.+)_layer\d+_evolution\.json$", name)
    if not match:
        raise ValueError(f"Unexpected per-layer filename format: {path}")
    return match.group(1)


def _parse_layer_idx_from_path(path: str, data: Dict[str, Any]) -> int:
    layer_idx = data.get("layer_idx")
    if layer_idx is not None:
        return int(layer_idx)
    match = re.search(r"_layer(\d+)_evolution\.json$", os.path.basename(path))
    if not match:
        raise ValueError(f"Could not infer layer index from: {path}")
    return int(match.group(1))


def _load_grouped_layer_results(input_dir: str) -> Dict[str, List[Tuple[int, Dict[str, Any]]]]:
    grouped: Dict[str, List[Tuple[int, Dict[str, Any]]]] = defaultdict(list)
    for path in _collect_layer_json_paths(input_dir):
        feature_name = _parse_feature_name_from_path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        layer_idx = _parse_layer_idx_from_path(path, data)
        grouped[feature_name].append((layer_idx, data))

    for feature_name in grouped:
        grouped[feature_name].sort(key=lambda item: item[0])
    return dict(grouped)


def _validate_required_modal_features(
    grouped_layer_entries: Dict[str, List[Tuple[int, Dict[str, Any]]]],
    required_features: List[str],
) -> None:
    missing_features = [
        feature_name for feature_name in required_features if feature_name not in grouped_layer_entries
    ]
    if missing_features:
        raise ValueError(
            f"Missing expected modal features in input-dir: {missing_features}"
        )


def _sort_feature_names(feature_names: List[str]) -> List[str]:
    return sorted(
        feature_names,
        key=lambda name: (FEATURE_ORDER_PRIORITY.get(name, len(FEATURE_ORDER_PRIORITY)), name),
    )


def _resolve_target_features(
    grouped_layer_entries: Dict[str, List[Tuple[int, Dict[str, Any]]]],
    explicit_features: Optional[List[str]],
) -> List[str]:
    if explicit_features:
        target_features = explicit_features
    else:
        target_features = list(grouped_layer_entries.keys())
    target_features = _sort_feature_names(target_features)
    _validate_required_modal_features(grouped_layer_entries, target_features)
    return target_features


def _get_single_layer_result(data: Dict[str, Any], layer: int) -> Dict[str, Any]:
    base_results = data.get("base_results", {})
    layer_results = base_results.get("layer_results", {})
    return layer_results.get(layer, layer_results.get(str(layer), {}))


def _get_full_stats(layer_result: Dict[str, Any]) -> Dict[str, Any]:
    full_stats = layer_result.get("full_stats", {})
    return full_stats if isinstance(full_stats, dict) else {}


def _get_base_vector_stats(
    full_stats: Dict[str, Any],
    base_vec: int,
) -> Dict[str, Any]:
    stats = full_stats.get(base_vec, full_stats.get(str(base_vec), {}))
    return stats if isinstance(stats, dict) else {}


def _extract_base_vectors_from_feature_pairs(items: Any) -> List[int]:
    base_vectors: List[int] = []
    if not isinstance(items, list):
        return base_vectors

    for item in items:
        if not isinstance(item, (list, tuple)) or not item:
            continue
        try:
            base_vectors.append(int(item[0]))
        except (TypeError, ValueError):
            continue
    return base_vectors


def _extract_base_vectors_from_scalar_list(items: Any) -> List[int]:
    base_vectors: List[int] = []
    if not isinstance(items, list):
        return base_vectors

    for item in items:
        try:
            base_vectors.append(int(item))
        except (TypeError, ValueError):
            continue
    return base_vectors


def _resolve_base_vector_source(
    layer_result: Dict[str, Any],
    source: str,
    selection_key: str,
) -> Dict[str, Any]:
    if source == "auto":
        source_candidates = [
            "intervention_features",
            "lasso",
            "elasticnet",
            "top_100_base_vectors_desc",
            "top_100_features",
            "top_features",
        ]
    else:
        source_candidates = [source]

    for candidate_source in source_candidates:
        if candidate_source == "top_features":
            base_vectors = _extract_base_vectors_from_feature_pairs(
                layer_result.get("top_features", [])
            )
            if selection_key == "top_3":
                base_vectors = base_vectors[:3]
        elif candidate_source == "top_100_features":
            base_vectors = _extract_base_vectors_from_feature_pairs(
                layer_result.get("top_100_features", [])
            )
        elif candidate_source == "top_100_base_vectors_desc":
            base_vectors = _extract_base_vectors_from_scalar_list(
                layer_result.get("top_100_base_vectors_desc", [])
            )
        elif candidate_source == "intervention_features":
            base_vectors = _extract_base_vectors_from_scalar_list(
                layer_result.get("intervention_features", [])
            )
        elif candidate_source == "lasso":
            lasso_selected = layer_result.get("lasso_selected_base_vectors", {})
            base_vectors = _extract_base_vectors_from_scalar_list(
                lasso_selected.get(selection_key, [])
                if isinstance(lasso_selected, dict)
                else []
            )
        elif candidate_source == "elasticnet":
            elasticnet_selected = layer_result.get("elasticnet_selected_base_vectors", {})
            base_vectors = _extract_base_vectors_from_scalar_list(
                elasticnet_selected.get(selection_key, [])
                if isinstance(elasticnet_selected, dict)
                else []
            )
        else:
            raise ValueError(f"Unsupported base vector source: {candidate_source}")

        if base_vectors:
            return {
                "status": "ready",
                "source": candidate_source,
                "selection_key": selection_key
                if candidate_source in {"lasso", "elasticnet"}
                else None,
                "base_vectors": base_vectors,
            }

    return {
        "status": "missing_candidates",
        "source": source,
        "selection_key": selection_key if source in {"lasso", "elasticnet"} else None,
        "base_vectors": [],
    }


def _compute_cross_layer_stats(evolution_data: Dict[str, Any]) -> Dict[str, Any]:
    stats = {
        "layer_overlap": {},
        "persistence_analysis": {},
        "emergence_analysis": {},
        "peak_layers": {},
    }

    layers = sorted(evolution_data["layer_progression"].keys())
    for idx in range(len(layers) - 1):
        layer1, layer2 = layers[idx], layers[idx + 1]
        layer1_vectors = set(evolution_data["layer_progression"][layer1]["top_base_vectors"])
        layer2_vectors = set(evolution_data["layer_progression"][layer2]["top_base_vectors"])
        overlap = len(layer1_vectors & layer2_vectors)
        union = len(layer1_vectors | layer2_vectors)
        stats["layer_overlap"][f"{layer1}->{layer2}"] = {
            "overlap_count": overlap,
            "jaccard_similarity": overlap / union if union > 0 else 0.0,
            "persistence_rate": overlap / len(layer1_vectors) if layer1_vectors else 0.0,
        }

    for base_vec, layer_stats in evolution_data["base_vector_evolution"].items():
        layers_present = sorted(layer_stats.keys())
        frc_scores = [float(layer_stats[layer].get("frc", 0.0)) for layer in layers_present]
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


def _build_evolution_results(
    layer_entries: List[Tuple[int, Dict[str, Any]]],
    source: str,
    selection_key: str,
) -> Dict[str, Any]:
    feature_file = None
    evolution_data = {
        "layer_progression": {},
        "base_vector_evolution": defaultdict(dict),
        "frc_evolution": {},
        "stability_analysis": {},
    }

    for layer, data in layer_entries:
        base_results = data.get("base_results", {})
        if feature_file is None:
            feature_file = base_results.get("feature_file")
        layer_result = _get_single_layer_result(data, layer)
        full_stats = _get_full_stats(layer_result)
        resolved = _resolve_base_vector_source(layer_result, source, selection_key)
        base_vectors = resolved["base_vectors"]
        if not base_vectors:
            continue

        frc_scores = []
        observed_base_vectors = []
        for base_vec in base_vectors:
            stats = _get_base_vector_stats(full_stats, base_vec)
            if not stats:
                continue
            observed_base_vectors.append(int(base_vec))
            frc_scores.append(float(stats.get("frc", 0.0)))
            evolution_data["base_vector_evolution"][int(base_vec)][layer] = stats

        if not observed_base_vectors:
            continue

        evolution_data["layer_progression"][layer] = {
            "top_base_vectors": observed_base_vectors,
            "top_frc_scores": frc_scores,
            "total_base_vectors": len(full_stats),
            "base_vector_source": resolved["source"],
            "selection_key": resolved["selection_key"],
        }

    return {
        "feature_file": feature_file,
        "layers_analyzed": [layer for layer, _ in layer_entries],
        "evolution_data": evolution_data,
        "cross_layer_stats": _compute_cross_layer_stats(evolution_data),
    }


def _group_vectors_by_feature_and_layer(
    grouped_layer_entries: Dict[str, List[Tuple[int, Dict[str, Any]]]],
    features: List[str],
    source: str,
    selection_key: str,
) -> Dict[int, Dict[str, List[int]]]:
    layer_feature_vectors: Dict[int, Dict[str, List[int]]] = defaultdict(dict)

    for feature_name in features:
        for layer, data in grouped_layer_entries[feature_name]:
            layer_result = _get_single_layer_result(data, layer)
            resolved = _resolve_base_vector_source(layer_result, source, selection_key)
            layer_feature_vectors[int(layer)][feature_name] = [
                int(base_vec) for base_vec in resolved["base_vectors"]
            ]

    return dict(sorted(layer_feature_vectors.items()))


def _compute_pairwise_overlap(list_a: List[int], list_b: List[int]) -> Dict[str, Any]:
    set_a = set(int(item) for item in list_a)
    set_b = set(int(item) for item in list_b)
    intersection = sorted(set_a & set_b)
    union = set_a | set_b
    return {
        "overlap_count": len(intersection),
        "intersection": intersection,
        "union_count": len(union),
        "jaccard_similarity": len(intersection) / len(union) if union else 0.0,
        "coverage_a": len(intersection) / len(set_a) if set_a else 0.0,
        "coverage_b": len(intersection) / len(set_b) if set_b else 0.0,
    }


def _summarize_group_overlap(feature_vectors: Dict[str, List[int]]) -> Dict[str, Any]:
    features = sorted(feature_vectors.keys())
    normalized_vectors = {
        feature: sorted({int(item) for item in feature_vectors.get(feature, [])})
        for feature in features
    }
    all_sets = {feature: set(values) for feature, values in normalized_vectors.items()}

    union_set = set().union(*all_sets.values()) if all_sets else set()
    intersection_set = set.intersection(*all_sets.values()) if all_sets else set()
    occurrence_counter = Counter()
    for values in all_sets.values():
        for base_vec in values:
            occurrence_counter[int(base_vec)] += 1

    duplicate_indices = sorted(
        base_vec for base_vec, count in occurrence_counter.items() if count >= 2
    )
    overlap_histogram = {
        str(count): int(sum(1 for curr_count in occurrence_counter.values() if curr_count == count))
        for count in sorted(set(occurrence_counter.values()))
    }

    pairwise = {}
    pairwise_jaccards = []
    pairwise_overlap_counts = []
    for feature_a, feature_b in itertools.combinations(features, 2):
        overlap = _compute_pairwise_overlap(
            normalized_vectors[feature_a],
            normalized_vectors[feature_b],
        )
        pairwise[f"{feature_a}__{feature_b}"] = overlap
        pairwise_jaccards.append(float(overlap["jaccard_similarity"]))
        pairwise_overlap_counts.append(int(overlap["overlap_count"]))

    return {
        "features": features,
        "feature_count": len(features),
        "union_count": len(union_set),
        "intersection_all_count": len(intersection_set),
        "intersection_all_indices": sorted(intersection_set),
        "duplicate_index_count": len(duplicate_indices),
        "overlap_histogram": overlap_histogram,
        "pairwise": pairwise,
        "mean_pairwise_jaccard": (
            sum(pairwise_jaccards) / len(pairwise_jaccards)
            if pairwise_jaccards
            else 0.0
        ),
        "mean_pairwise_overlap_count": (
            sum(pairwise_overlap_counts) / len(pairwise_overlap_counts)
            if pairwise_overlap_counts
            else 0.0
        ),
    }


def _build_modal_overlap_summary(
    grouped_layer_entries: Dict[str, List[Tuple[int, Dict[str, Any]]]],
    source: str,
    selection_key: str,
    target_features: List[str],
) -> Dict[str, Any]:
    layer_feature_vectors = _group_vectors_by_feature_and_layer(
        grouped_layer_entries,
        target_features,
        source,
        selection_key,
    )

    layers = sorted(layer_feature_vectors.keys())
    layer_overlap = {}
    for layer in layers:
        feature_vectors = layer_feature_vectors[layer]
        layer_overlap[str(layer)] = _summarize_group_overlap(feature_vectors)

    return {
        "target_features": target_features,
        "candidate_source": source,
        "selection_key": selection_key,
        "layers": layers,
        "layer_overlap": layer_overlap,
    }


def _build_jaccard_distance_matrix(
    feature_vectors: Dict[str, List[int]],
    features: List[str],
) -> Tuple[List[List[float]], Dict[str, Dict[str, List[int]]]]:
    distance_matrix: List[List[float]] = []
    pairwise_intersections: Dict[str, Dict[str, List[int]]] = {
        feature: {} for feature in features
    }

    for feature_a in features:
        row = []
        for feature_b in features:
            overlap = _compute_pairwise_overlap(
                feature_vectors.get(feature_a, []),
                feature_vectors.get(feature_b, []),
            )
            row.append(1.0 - float(overlap["jaccard_similarity"]))
            pairwise_intersections[feature_a][feature_b] = list(overlap["intersection"])
        distance_matrix.append(row)

    return distance_matrix, pairwise_intersections


def _labels_to_clusters(features: List[str], cluster_labels: List[int]) -> Dict[str, List[str]]:
    clusters: Dict[int, List[str]] = defaultdict(list)
    for feature_name, cluster_label in zip(features, cluster_labels):
        clusters[int(cluster_label)].append(feature_name)
    return {
        str(cluster_id): sorted(cluster_features)
        for cluster_id, cluster_features in sorted(clusters.items())
    }


def _evaluate_realis_vs_irrealis(
    features: List[str],
    cluster_labels: List[int],
    realis_feature: str = "know",
) -> Dict[str, Any]:
    if realis_feature not in features:
        return {
            "evaluated": False,
            "reason": f"Missing realis feature: {realis_feature}",
            "passed": False,
        }

    irrealis_features = [feature for feature in features if feature != realis_feature]
    if not irrealis_features:
        return {
            "evaluated": False,
            "reason": "No irrealis features available",
            "passed": False,
        }

    cluster_map = dict(zip(features, cluster_labels))
    realis_cluster = cluster_map[realis_feature]
    irrealis_clusters = {cluster_map[feature] for feature in irrealis_features}
    passed = realis_cluster not in irrealis_clusters and len(irrealis_clusters) == 1
    return {
        "evaluated": True,
        "realis_feature": realis_feature,
        "irrealis_features": irrealis_features,
        "realis_cluster": int(realis_cluster),
        "irrealis_clusters": sorted(int(cluster_id) for cluster_id in irrealis_clusters),
        "passed": passed,
    }


def _evaluate_suppose_vs_modal_verbs(
    irrealis_features: List[str],
    cluster_labels: List[int],
    suppose_feature: str = "suppose",
) -> Dict[str, Any]:
    if suppose_feature not in irrealis_features:
        return {
            "evaluated": False,
            "reason": f"Missing irrealis split anchor: {suppose_feature}",
            "passed": False,
        }

    modal_features = [feature for feature in irrealis_features if feature != suppose_feature]
    if not modal_features:
        return {
            "evaluated": False,
            "reason": "No modal verb features available inside irrealis set",
            "passed": False,
        }

    cluster_map = dict(zip(irrealis_features, cluster_labels))
    suppose_cluster = cluster_map[suppose_feature]
    modal_clusters = {cluster_map[feature] for feature in modal_features}
    passed = suppose_cluster not in modal_clusters and len(modal_clusters) == 1
    return {
        "evaluated": True,
        "suppose_feature": suppose_feature,
        "modal_features": modal_features,
        "suppose_cluster": int(suppose_cluster),
        "modal_clusters": sorted(int(cluster_id) for cluster_id in modal_clusters),
        "passed": passed,
    }


def _cluster_features_from_distance_matrix(
    feature_vectors: Dict[str, List[int]],
    features: List[str],
    linkage_method: str,
    flat_k: int,
) -> Dict[str, Any]:
    if len(features) < 2:
        return {
            "features": features,
            "distance_matrix": [],
            "pairwise_intersections": {},
            "linkage_method": linkage_method,
            "linkage_matrix": [],
            "leaf_order": features,
            "flat_clusters": {},
            "cluster_labels": [],
        }

    import numpy as np
    from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
    from scipy.spatial.distance import squareform

    distance_matrix, pairwise_intersections = _build_jaccard_distance_matrix(
        feature_vectors,
        features,
    )
    condensed = squareform(np.array(distance_matrix, dtype=float), checks=False)
    linkage_matrix = linkage(condensed, method=linkage_method)
    cluster_labels = fcluster(linkage_matrix, t=flat_k, criterion="maxclust").tolist()
    leaf_indices = leaves_list(linkage_matrix).tolist()

    return {
        "features": features,
        "distance_matrix": distance_matrix,
        "pairwise_intersections": pairwise_intersections,
        "linkage_method": linkage_method,
        "linkage_matrix": linkage_matrix.tolist(),
        "leaf_order": [features[idx] for idx in leaf_indices],
        "flat_clusters": _labels_to_clusters(features, cluster_labels),
        "cluster_labels": [int(label) for label in cluster_labels],
    }


def _build_hierarchical_clustering_summary(
    grouped_layer_entries: Dict[str, List[Tuple[int, Dict[str, Any]]]],
    source: str,
    selection_key: str,
    target_features: List[str],
) -> Dict[str, Any]:
    layer_feature_vectors = _group_vectors_by_feature_and_layer(
        grouped_layer_entries,
        target_features,
        source,
        selection_key,
    )

    linkage_methods = ("average", "complete", "single")
    layers = sorted(layer_feature_vectors.keys())
    layer_clusterings: Dict[str, Any] = {}

    for layer in layers:
        feature_vectors = layer_feature_vectors[layer]
        present_features = _sort_feature_names(
            [feature for feature in target_features if feature in feature_vectors]
        )
        irrealis_features = [feature for feature in present_features if feature != "know"]

        linkage_results: Dict[str, Any] = {}
        for linkage_method in linkage_methods:
            all_feature_result = _cluster_features_from_distance_matrix(
                feature_vectors,
                present_features,
                linkage_method,
                flat_k=2,
            )
            irrealis_result = _cluster_features_from_distance_matrix(
                feature_vectors,
                irrealis_features,
                linkage_method,
                flat_k=2,
            )

            linkage_results[linkage_method] = {
                "all_features": {
                    key: value
                    for key, value in all_feature_result.items()
                    if key != "cluster_labels"
                },
                "irrealis_only": {
                    key: value
                    for key, value in irrealis_result.items()
                    if key != "cluster_labels"
                },
                "evaluations": {
                    "realis_vs_irrealis": _evaluate_realis_vs_irrealis(
                        present_features,
                        all_feature_result["cluster_labels"],
                    ),
                    "suppose_vs_modal_verbs_within_irrealis": _evaluate_suppose_vs_modal_verbs(
                        irrealis_features,
                        irrealis_result["cluster_labels"],
                    ),
                },
            }

        layer_clusterings[str(layer)] = {
            "present_features": present_features,
            "linkage_methods": linkage_results,
        }

    return {
        "target_features": target_features,
        "candidate_source": source,
        "selection_key": selection_key,
        "layers": layers,
        "layer_clusterings": layer_clusterings,
    }


def _save_modal_overlap_summary(
    grouped_layer_entries: Dict[str, List[Tuple[int, Dict[str, Any]]]],
    output_dir: str,
    target_features: List[str],
) -> None:
    overlap_summary = {
        "target_features": target_features,
        "comparisons": {},
    }

    for comparison_name, (source, selection_key) in DEFAULT_OVERLAP_SELECTIONS.items():
        overlap_summary["comparisons"][comparison_name] = _build_modal_overlap_summary(
            grouped_layer_entries,
            source,
            selection_key,
            target_features,
        )

    output_path = os.path.join(output_dir, "modal_layer_overlap_summary.json")
    save_json_results(overlap_summary, output_path)
    print(f"Saved modal overlap summary: {output_path}")


def _save_hierarchical_dendrograms(
    clustering_summary: Dict[str, Any],
    output_dir: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required to generate hierarchical clustering figures."
        ) from exc

    import numpy as np
    from scipy.cluster.hierarchy import dendrogram

    figures_dir = os.path.join(output_dir, "hierarchical_clustering_figures")
    os.makedirs(figures_dir, exist_ok=True)

    for comparison_name, comparison_summary in clustering_summary["comparisons"].items():
        for layer, layer_summary in comparison_summary["layer_clusterings"].items():
            for linkage_method, linkage_summary in layer_summary["linkage_methods"].items():
                for scope_name in ("all_features", "irrealis_only"):
                    scope_summary = linkage_summary[scope_name]
                    features = scope_summary.get("features", [])
                    linkage_matrix = scope_summary.get("linkage_matrix", [])
                    if len(features) < 2 or not linkage_matrix:
                        continue

                    plt.figure(figsize=(10, 6))
                    dendrogram(
                        np.array(linkage_matrix, dtype=float),
                        labels=features,
                        leaf_rotation=45,
                        leaf_font_size=10,
                    )
                    plt.title(
                        f"Hierarchical Clustering - {comparison_name} - layer {layer} - {linkage_method} - {scope_name}"
                    )
                    plt.ylabel("Jaccard Distance")
                    plt.tight_layout()
                    output_path = os.path.join(
                        figures_dir,
                        f"{comparison_name}_layer{layer}_{linkage_method}_{scope_name}.png",
                    )
                    plt.savefig(output_path, dpi=300, bbox_inches="tight")
                    plt.close()
                    print(f"Saved hierarchical clustering figure: {output_path}")


def _save_hierarchical_clustering_summary(
    grouped_layer_entries: Dict[str, List[Tuple[int, Dict[str, Any]]]],
    output_dir: str,
    target_features: List[str],
) -> None:
    clustering_summary = {
        "target_features": target_features,
        "comparisons": {},
    }

    for comparison_name, (source, selection_key) in DEFAULT_OVERLAP_SELECTIONS.items():
        clustering_summary["comparisons"][comparison_name] = (
            _build_hierarchical_clustering_summary(
                grouped_layer_entries,
                source,
                selection_key,
                target_features,
            )
        )

    output_path = os.path.join(output_dir, "hierarchical_clustering_summary.json")
    save_json_results(clustering_summary, output_path)
    print(f"Saved hierarchical clustering summary: {output_path}")
    _save_hierarchical_dendrograms(clustering_summary, output_dir)


def _save_evolution_plots_for_all_features(
    grouped_layer_entries: Dict[str, List[Tuple[int, Dict[str, Any]]]],
    output_dir: str,
    metric: str,
    target_features: List[str],
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "matplotlib is required to generate plots."
        ) from exc

    for comparison_name, (source, selection_key) in DEFAULT_OVERLAP_SELECTIONS.items():
        for feature_name in target_features:
            layer_entries = grouped_layer_entries[feature_name]
            evolution_results = _build_evolution_results(layer_entries, source, selection_key)
            evolution_data = evolution_results["evolution_data"]

            plt.figure(figsize=(12, 8))
            for base_vec, layer_stats in evolution_data["base_vector_evolution"].items():
                if len(layer_stats) < 2:
                    continue
                x_layers = sorted(layer_stats.keys())
                y_values = [float(layer_stats[layer].get(metric, 0.0)) for layer in x_layers]
                plt.plot(x_layers, y_values, marker="o", alpha=0.7, label=f"BV {base_vec}")

            plt.xlabel("Layer Index")
            plt.ylabel(f"{metric.upper()} Score")
            plt.title(
                f"Feature Evolution Across Layers - {feature_name} - {comparison_name} - {metric.upper()}"
            )
            plt.grid(True, alpha=0.3)
            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.tight_layout()
            output_path = os.path.join(output_dir, f"{feature_name}_{comparison_name}_{metric}.png")
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"Saved evolution plot: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Postprocess per-layer crosslayer outputs for modal/suppose/know runs."
        )
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing per-layer JSONs for all target features.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Default: same as --input-dir.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help=(
            "Generate cross-layer plots for all target features across top3, top100, "
            "Lasso top10/top20/top50/top100, and ElasticNet top10/top20/top50/top100."
        ),
    )
    parser.add_argument(
        "--plot-metric",
        default="frc",
        help="Metric key for evolution plot (default: frc).",
    )
    parser.add_argument(
        "--compare-modal-overlap",
        action="store_true",
        help=(
            "Compare layer-wise index overlap across the target feature set for top3, top100, "
            "Lasso, and ElasticNet."
        ),
    )
    parser.add_argument(
        "--cluster-hierarchical",
        action="store_true",
        help=(
            "Run hierarchical clustering from layer-wise Jaccard distances using "
            "average/complete/single linkage, and evaluate know-vs-others plus "
            "suppose-vs-modal-verb separation."
        ),
    )
    parser.add_argument(
        "--target-features",
        nargs="+",
        default=None,
        help=(
            "Optional explicit feature list. Default: detect all feature names "
            "present under --input-dir."
        ),
    )

    args = parser.parse_args()
    if not args.plot and not args.compare_modal_overlap and not args.cluster_hierarchical:
        raise ValueError(
            "Specify at least one of --plot, --compare-modal-overlap, or --cluster-hierarchical."
        )

    output_dir = args.output_dir or args.input_dir
    os.makedirs(output_dir, exist_ok=True)

    grouped_layer_entries = _load_grouped_layer_results(args.input_dir)
    target_features = _resolve_target_features(
        grouped_layer_entries,
        args.target_features,
    )

    if args.compare_modal_overlap:
        _save_modal_overlap_summary(
            grouped_layer_entries,
            output_dir,
            target_features,
        )

    if args.cluster_hierarchical:
        _save_hierarchical_clustering_summary(
            grouped_layer_entries,
            output_dir,
            target_features,
        )

    if args.plot:
        _save_evolution_plots_for_all_features(
            grouped_layer_entries,
            output_dir,
            args.plot_metric,
            target_features,
        )


if __name__ == "__main__":
    main()
