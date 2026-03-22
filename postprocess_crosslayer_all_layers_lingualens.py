#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import re
import time
from glob import glob
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.linear_model import ElasticNetCV, LassoCV

from analyzer_lingualens import TrainSaeLinguisticAnalyzer
from LinguaLens.lingualens.utils import load_text_data, save_json_results, validate_layer_indices
from model import normalize_activation

LASSO_SELECTION_COUNTS = (10, 20, 50, 100)
PER_LAYER_TOP_COUNTS = (3, 5, 10)


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


def _get_single_layer_result(data: Dict[str, Any], layer: int) -> Dict[str, Any]:
    base_results = data.get("base_results", {})
    layer_results = base_results.get("layer_results", {})
    return layer_results.get(layer, layer_results.get(str(layer), {}))


def _get_full_stats(layer_result: Dict[str, Any]) -> Dict[str, Any]:
    full_stats = layer_result.get("full_stats", {})
    return full_stats if isinstance(full_stats, dict) else {}


def _parse_torch_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")
    return mapping[dtype_name]


def _parse_top_counts(top_counts_arg: str) -> List[int]:
    top_counts = [int(item.strip()) for item in top_counts_arg.split(",") if item.strip()]
    if not top_counts:
        raise ValueError("Specify at least one top-count value.")
    if any(top_count <= 0 for top_count in top_counts):
        raise ValueError(f"Top counts must be positive integers: {top_counts}")
    return sorted(set(top_counts))


def _build_candidate_sets(
    layer_entries: List[Tuple[int, Dict[str, Any]]],
    top_counts: List[int],
) -> Tuple[str, List[int], Dict[int, Dict[str, Any]], Dict[str, Any]]:
    feature_file: Optional[str] = None
    layers = validate_layer_indices([layer for layer, _ in layer_entries])
    top_count_candidates: Dict[int, Dict[str, Any]] = {}
    source_summary: Dict[str, Any] = {
        "layers": layers,
        "per_layer_top_counts": top_counts,
        "layer_feature_summaries": {},
    }

    for top_count in top_counts:
        candidate_features: List[Dict[str, Any]] = []
        selected_by_layer: Dict[int, List[int]] = {}

        for layer, data in layer_entries:
            base_results = data.get("base_results", {})
            if feature_file is None:
                feature_file = base_results.get("feature_file")

            layer_result = _get_single_layer_result(data, layer)
            full_stats = _get_full_stats(layer_result)
            top_features = layer_result.get("top_100_features") or layer_result.get("top_features") or []
            if len(top_features) < top_count:
                raise ValueError(
                    f"Layer {layer} has only {len(top_features)} top features, cannot build top_{top_count}."
                )

            selected_base_vectors: List[int] = []
            for rank_in_layer, feature_pair in enumerate(top_features[:top_count], start=1):
                if not isinstance(feature_pair, (list, tuple)) or len(feature_pair) < 2:
                    raise ValueError(
                        f"Invalid feature pair in layer {layer}: {feature_pair!r}"
                    )
                base_vector = int(feature_pair[0])
                frc = float(feature_pair[1])
                stats = full_stats.get(base_vector, full_stats.get(str(base_vector), {}))
                feature_id = f"layer{layer}_bv{base_vector}"
                candidate_features.append(
                    {
                        "feature_id": feature_id,
                        "layer": int(layer),
                        "base_vector": base_vector,
                        "rank_within_layer": int(rank_in_layer),
                        "frc": frc,
                        "ps": float(stats.get("ps", 0.0)),
                        "pn": float(stats.get("pn", 0.0)),
                        "avg_max_activation": float(stats.get("avg_max_activation", 0.0)),
                    }
                )
                selected_base_vectors.append(base_vector)

            selected_by_layer[int(layer)] = selected_base_vectors
            source_summary["layer_feature_summaries"][str(layer)] = {
                "available_top_features": len(top_features),
                "source": "top_100_features" if layer_result.get("top_100_features") else "top_features",
            }

        top_count_candidates[top_count] = {
            "candidate_features": candidate_features,
            "selected_by_layer": selected_by_layer,
        }

    if feature_file is None:
        raise ValueError("Could not infer feature_file from per-layer crosslayer outputs.")

    return feature_file, layers, top_count_candidates, source_summary


class TrainSaeAllLayersAnalyzer:
    def __init__(
        self,
        model_path: str,
        sae_path_template: str,
        device: Optional[str],
        k: int,
        normalization: str,
        batch_size: int,
        torch_dtype: torch.dtype,
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
        self.normalization = normalization
        self.batch_size = int(batch_size)

    def collect_sentence_feature_rows(
        self,
        feature_file: str,
        layers: List[int],
        selected_by_layer: Dict[int, List[int]],
    ) -> List[Dict[str, Any]]:
        lines = load_text_data(feature_file)
        model = self.analyzer._load_base_model()
        runtimes = {
            int(layer): self.analyzer._get_sae_model(int(layer))
            for layer in layers
        }

        sentence_feature_rows: List[Dict[str, Any]] = []
        total_batches = max(1, (len(lines) + self.batch_size - 1) // self.batch_size)
        log_interval = max(1, total_batches // 10)

        for batch_idx, start in enumerate(range(0, len(lines), self.batch_size), start=1):
            if batch_idx == 1 or batch_idx == total_batches or batch_idx % log_interval == 0:
                print(
                    f"[all-layers] batch {batch_idx}/{total_batches} "
                    f"(examples {start + 1}-{min(start + self.batch_size, len(lines))}/{len(lines)})"
                )

            batch_lines = lines[start : start + self.batch_size]
            enc = self.analyzer.tokenizer(
                batch_lines,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            input_ids = enc["input_ids"].to(self.analyzer.device)
            attention_mask = enc.get("attention_mask")
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
            attention_mask = attention_mask.to(self.analyzer.device)

            batch_rows = []
            for row_idx in range(input_ids.size(0)):
                sentence_id = start + row_idx + 1
                batch_rows.append(
                    {
                        "sentence_id": int(sentence_id),
                        "label": 1 if sentence_id % 2 == 1 else 0,
                        "line_type": "original" if sentence_id % 2 == 1 else "minimal_pair",
                        "feature_activations": {},
                    }
                )

            with torch.no_grad():
                _ = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )

            token_mask = attention_mask[:, 1:]
            for layer in layers:
                runtime = runtimes[int(layer)]
                sae = runtime["sae"]
                hook = runtime["hook"]
                layer_candidates = selected_by_layer.get(int(layer), [])
                layer_candidate_set = set(int(base_vector) for base_vector in layer_candidates)
                if not layer_candidate_set:
                    hook.output = None
                    continue

                acts = hook.output
                if isinstance(acts, tuple):
                    acts = acts[0]
                if acts.dim() == 2:
                    acts = acts.unsqueeze(0)
                acts = acts[:, 1:, :]

                for row_idx in range(input_ids.size(0)):
                    valid_len = int(token_mask[row_idx].sum().item())
                    layer_activation_map = {
                        f"layer{layer}_bv{base_vector}": 0.0
                        for base_vector in layer_candidates
                    }
                    if valid_len > 0:
                        activation = acts[row_idx, :valid_len, :]
                        activation = normalize_activation(activation, self.normalization)
                        activation = activation.to(self.analyzer.device)

                        with torch.no_grad():
                            out = sae(activation)

                        top_indices = out.latent_indices.detach().cpu().tolist()
                        top_acts = out.latent_acts.detach().cpu().tolist()
                        for token_indices, token_values in zip(top_indices, top_acts):
                            for base_vector, value in zip(token_indices, token_values):
                                if value <= 0 or int(base_vector) not in layer_candidate_set:
                                    continue
                                feature_id = f"layer{layer}_bv{int(base_vector)}"
                                current_value = layer_activation_map[feature_id]
                                if float(value) > current_value:
                                    layer_activation_map[feature_id] = float(value)

                    batch_rows[row_idx]["feature_activations"].update(layer_activation_map)

                hook.output = None

            sentence_feature_rows.extend(batch_rows)

        return sentence_feature_rows

    def clear_cache(self) -> None:
        self.analyzer.clear_cache()


def _fit_cross_layer_selector(
    sentence_feature_rows: List[Dict[str, Any]],
    candidate_features: List[Dict[str, Any]],
) -> Dict[str, Any]:
    def _standardize_matrix(matrix: np.ndarray) -> np.ndarray:
        feature_means = matrix.mean(axis=0, dtype=np.float64)
        feature_stds = matrix.std(axis=0, dtype=np.float64)
        feature_stds[feature_stds < 1e-6] = 1.0
        standardized = (matrix - feature_means) / feature_stds
        return standardized.astype(np.float64, copy=False)

    def _feature_ref(feature: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "feature_id": feature["feature_id"],
            "layer": int(feature["layer"]),
            "base_vector": int(feature["base_vector"]),
        }

    def _build_ranked_feature_weights(
        ranked_indices: List[int],
        coefficients: np.ndarray,
        scores: np.ndarray,
        score_name: str,
    ) -> List[Dict[str, Any]]:
        ranked_feature_weights: List[Dict[str, Any]] = []
        for rank_idx in ranked_indices:
            if rank_idx >= len(candidate_features):
                continue
            feature = candidate_features[rank_idx]
            ranked_feature_weights.append(
                {
                    "feature_id": feature["feature_id"],
                    "layer": int(feature["layer"]),
                    "base_vector": int(feature["base_vector"]),
                    "rank_within_layer": int(feature["rank_within_layer"]),
                    "weight": float(coefficients[rank_idx]),
                    "abs_weight": float(abs(coefficients[rank_idx])),
                    score_name: float(scores[rank_idx]),
                    "frc": float(feature["frc"]),
                    "ps": float(feature["ps"]),
                    "pn": float(feature["pn"]),
                    "avg_max_activation": float(feature["avg_max_activation"]),
                }
            )
        return ranked_feature_weights

    def _build_topn_selections(
        ranked_indices: List[int],
        ranked_feature_weights: List[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], Dict[str, List[Dict[str, Any]]]]:
        selections_by_count: Dict[str, Any] = {}
        selected_features_by_count: Dict[str, List[Dict[str, Any]]] = {}
        available_count = len(ranked_indices)
        for selection_count in LASSO_SELECTION_COUNTS:
            selection_key = f"top_{selection_count}"
            capped_count = min(selection_count, available_count)
            selected_rank_indices = ranked_indices[:capped_count]
            selected_features = [_feature_ref(candidate_features[idx]) for idx in selected_rank_indices]
            selected_features_by_count[selection_key] = selected_features
            selections_by_count[selection_key] = {
                "selection_count_requested": int(selection_count),
                "selection_count_returned": int(capped_count),
                "selected_features": selected_features,
                "feature_weights": ranked_feature_weights[:capped_count],
            }
        return selections_by_count, selected_features_by_count

    candidate_feature_ids = [feature["feature_id"] for feature in candidate_features]
    labels = np.asarray(
        [float(row.get("label", 0)) for row in sentence_feature_rows],
        dtype=np.float32,
    )
    feature_matrix = np.zeros(
        (len(sentence_feature_rows), len(candidate_feature_ids)),
        dtype=np.float32,
    )

    for row_idx, row in enumerate(sentence_feature_rows):
        feature_activations = row.get("feature_activations", {})
        for col_idx, feature_id in enumerate(candidate_feature_ids):
            feature_matrix[row_idx, col_idx] = float(
                feature_activations.get(feature_id, 0.0)
            )

    coefficients = np.zeros(len(candidate_feature_ids), dtype=np.float32)
    stability_scores = np.zeros(len(candidate_feature_ids), dtype=np.float32)
    elasticnet_coefficients = np.zeros(len(candidate_feature_ids), dtype=np.float32)
    metadata: Dict[str, Any] = {
        "status": "not_run",
        "method": "lasso_stability_selection",
        "candidate_features": [_feature_ref(feature) for feature in candidate_features],
        "num_examples": int(len(sentence_feature_rows)),
        "label_mapping": {
            "1": "odd_lines_original_text",
            "0": "even_lines_minimal_pair",
        },
    }
    elasticnet_metadata: Dict[str, Any] = {
        "status": "not_run",
        "method": "elasticnet_cv",
        "candidate_features": [_feature_ref(feature) for feature in candidate_features],
        "num_examples": int(len(sentence_feature_rows)),
        "label_mapping": {
            "1": "odd_lines_original_text",
            "0": "even_lines_minimal_pair",
        },
    }

    can_fit = (
        len(candidate_feature_ids) > 0
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
        non_zero_counts = np.zeros(len(candidate_feature_ids), dtype=np.int32)

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
        if len(candidate_feature_ids) == 0:
            metadata["reason"] = "no_candidate_features"
            elasticnet_metadata["reason"] = "no_candidate_features"
        elif len(sentence_feature_rows) < 2:
            metadata["reason"] = "not_enough_examples"
            elasticnet_metadata["reason"] = "not_enough_examples"
        else:
            metadata["reason"] = "single_label_only"
            elasticnet_metadata["reason"] = "single_label_only"

    ranked_indices = sorted(
        range(len(candidate_feature_ids)),
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
    selections_by_count, selected_features_by_count = _build_topn_selections(
        ranked_indices=ranked_indices,
        ranked_feature_weights=feature_weights,
    )

    elasticnet_ranked_indices = sorted(
        range(len(candidate_feature_ids)),
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
        elasticnet_selected_features_by_count,
    ) = _build_topn_selections(
        ranked_indices=elasticnet_ranked_indices,
        ranked_feature_weights=elasticnet_feature_weights,
    )

    metadata["selection_counts"] = [int(count) for count in LASSO_SELECTION_COUNTS]
    elasticnet_metadata["selection_counts"] = [int(count) for count in LASSO_SELECTION_COUNTS]

    default_selection_key = "top_100"
    return {
        "candidate_features": candidate_features,
        "num_candidate_features": int(len(candidate_features)),
        "selected_features": selected_features_by_count.get(default_selection_key, []),
        "intervention_features": selected_features_by_count.get(default_selection_key, []),
        "feature_weights": feature_weights,
        "intervention_feature_weights": feature_weights,
        "metadata": metadata,
        "stability_selection": metadata,
        "selections_by_count": selections_by_count,
        "lasso_selections": selections_by_count,
        "selected_features_by_count": selected_features_by_count,
        "lasso_selected_features": selected_features_by_count,
        "elasticnet_feature_weights": elasticnet_feature_weights,
        "elasticnet_metadata": elasticnet_metadata,
        "elasticnet_selection": elasticnet_metadata,
        "elasticnet_selections_by_count": elasticnet_selections_by_count,
        "elasticnet_selections": elasticnet_selections_by_count,
        "elasticnet_selected_features_by_count": elasticnet_selected_features_by_count,
        "elasticnet_selected_features": elasticnet_selected_features_by_count,
    }


def analyze_all_layers(
    model_path: str,
    sae_path_template: str,
    input_dir: str,
    output_dir: str,
    feature_file: Optional[str],
    device: Optional[str],
    k: int,
    normalization: str,
    batch_size: int,
    torch_dtype: torch.dtype,
    top_counts: List[int],
) -> str:
    start_time = time.perf_counter()
    layer_entries = _load_layer_results(input_dir)
    inferred_feature_file, layers, top_count_candidates, source_summary = _build_candidate_sets(
        layer_entries,
        top_counts,
    )
    feature_file = feature_file or inferred_feature_file
    if feature_file is None:
        raise ValueError("feature_file is required or must be present in per-layer outputs.")

    os.makedirs(output_dir, exist_ok=True)
    feature_name = _infer_feature_name(_collect_layer_json_paths(input_dir))
    max_top_count = max(top_counts)
    selected_by_layer_max = top_count_candidates[max_top_count]["selected_by_layer"]

    print(f"[all-layers] feature file: {feature_file}")
    print(f"[all-layers] layers: {','.join(map(str, layers))}")
    print(f"[all-layers] per-layer top counts: {','.join(map(str, top_counts))}")
    print("[all-layers] phase 1/3: collecting sentence activations across layers")

    analyzer = TrainSaeAllLayersAnalyzer(
        model_path=model_path,
        sae_path_template=sae_path_template,
        device=device,
        k=k,
        normalization=normalization,
        batch_size=batch_size,
        torch_dtype=torch_dtype,
    )

    try:
        sentence_feature_rows = analyzer.collect_sentence_feature_rows(
            feature_file=feature_file,
            layers=layers,
            selected_by_layer=selected_by_layer_max,
        )
    finally:
        analyzer.clear_cache()

    print("[all-layers] phase 2/3: fitting cross-layer Lasso and ElasticNet")
    all_layers_results: Dict[str, Any] = {}
    for top_count in top_counts:
        selection_key = f"top_{top_count}_per_layer"
        candidate_features = top_count_candidates[top_count]["candidate_features"]
        result = _fit_cross_layer_selector(
            sentence_feature_rows=sentence_feature_rows,
            candidate_features=candidate_features,
        )
        result["per_layer_top_count"] = int(top_count)
        all_layers_results[selection_key] = result
        print(
            f"[all-layers] completed {selection_key}: "
            f"candidates={len(candidate_features)}, "
            f"elapsed={time.perf_counter() - start_time:.1f}s"
        )

    print("[all-layers] phase 3/3: saving results")
    output = {
        "feature_file": feature_file,
        "layers_analyzed": layers,
        "input_dir": input_dir,
        "top_counts_per_layer": top_counts,
        "total_examples": len(sentence_feature_rows),
        "source_summary": source_summary,
        "all_layers_results": all_layers_results,
    }

    output_path = os.path.join(output_dir, f"{feature_name}_layers_all_evolution.json")
    save_json_results(output, output_path)
    print(
        f"[all-layers] done: output={output_path}, "
        f"elapsed={time.perf_counter() - start_time:.1f}s"
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run all-layer Lasso and ElasticNet selection from per-layer "
            "crosslayer outputs."
        )
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Path or HF name of the base LLM.",
    )
    parser.add_argument(
        "--sae-path-template",
        required=True,
        help="SAE checkpoint template, e.g. /path/to/sae_layer{}.pth",
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing *_layer*_evolution.json files for one feature.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save *_layers_all_evolution.json.",
    )
    parser.add_argument(
        "--feature-file",
        default=None,
        help="Optional feature text file override. Default: inferred from per-layer outputs.",
    )
    parser.add_argument(
        "--top-counts",
        default=",".join(str(count) for count in PER_LAYER_TOP_COUNTS),
        help="Comma-separated per-layer top-N counts to evaluate (default: 3,5,10).",
    )
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

    args = parser.parse_args()
    top_counts = _parse_top_counts(args.top_counts)
    analyze_all_layers(
        model_path=args.model_path,
        sae_path_template=args.sae_path_template,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        feature_file=args.feature_file,
        device=args.device,
        k=args.k,
        normalization=args.normalization,
        batch_size=args.batch_size,
        torch_dtype=_parse_torch_dtype(args.torch_dtype),
        top_counts=top_counts,
    )


if __name__ == "__main__":
    main()
