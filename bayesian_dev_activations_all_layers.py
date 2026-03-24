#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import re
from glob import glob
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from LinguaLens.lingualens.utils import save_json_results

DEFAULT_EXCLUDED_TARGETS = {"know"}
COMPARISON_SPECS = {
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


def _parse_csv_list(raw_value: Optional[str]) -> List[str]:
    if not raw_value:
        return []
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def _discover_target_directories(
    input_dir: str,
    include_targets: Optional[List[str]],
    exclude_targets: Sequence[str],
) -> Dict[str, Dict[str, str]]:
    include_set = set(include_targets or [])
    exclude_set = set(exclude_targets)
    discovered: Dict[str, Dict[str, str]] = {}

    for dev_path in sorted(glob(os.path.join(input_dir, "*", "data", "*_dev.txt"))):
        target_name = os.path.basename(dev_path)[: -len("_dev.txt")]
        if target_name in exclude_set:
            continue
        if include_set and target_name not in include_set:
            continue
        target_root = os.path.dirname(os.path.dirname(dev_path))
        crosslayer_dir = os.path.join(target_root, "crosslayer")
        if not os.path.isdir(crosslayer_dir):
            raise FileNotFoundError(f"Missing crosslayer directory for {target_name}: {crosslayer_dir}")
        discovered[target_name] = {
            "dev_path": dev_path,
            "crosslayer_dir": crosslayer_dir,
        }

    if include_set:
        missing_targets = sorted(include_set - set(discovered.keys()))
        if missing_targets:
            raise FileNotFoundError(
                f"Could not find target directories under {input_dir}: {missing_targets}"
            )

    if not discovered:
        raise FileNotFoundError(f"No eligible target/data/*_dev.txt files found under {input_dir}.")

    return discovered


def _discover_activation_files(activations_dir: str) -> Dict[int, str]:
    activation_files: Dict[int, str] = {}
    pattern = os.path.join(activations_dir, "layer*_dev_activations.parquet")
    for path in sorted(glob(pattern)):
        match = re.search(r"layer(\d+)_dev_activations\.parquet$", os.path.basename(path))
        if match:
            activation_files[int(match.group(1))] = path
    if not activation_files:
        raise FileNotFoundError(f"No layer activation parquet files found under {activations_dir}")
    return activation_files


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_layer_result(data: Dict[str, Any], layer_idx: int) -> Dict[str, Any]:
    base_results = data.get("base_results", {})
    layer_results = base_results.get("layer_results", {})
    return layer_results.get(layer_idx, layer_results.get(str(layer_idx), {}))


def _extract_feature_pairs(items: Any) -> List[int]:
    if not isinstance(items, list):
        return []
    out: List[int] = []
    for item in items:
        if isinstance(item, (list, tuple)) and item:
            try:
                out.append(int(item[0]))
            except (TypeError, ValueError):
                continue
    return out


def _extract_scalar_list(items: Any) -> List[int]:
    if not isinstance(items, list):
        return []
    out: List[int] = []
    for item in items:
        try:
            out.append(int(item))
        except (TypeError, ValueError):
            continue
    return out


def _resolve_selected_indices(layer_result: Dict[str, Any], source: str, selection_key: str) -> List[int]:
    if source == "top_features":
        indices = _extract_feature_pairs(layer_result.get("top_features", []))
        if selection_key == "top_3":
            return indices[:3]
        return indices
    if source == "top_100_features":
        return _extract_feature_pairs(layer_result.get("top_100_features", []))
    if source == "lasso":
        selected = layer_result.get("lasso_selected_base_vectors", {})
        return _extract_scalar_list(selected.get(selection_key, []) if isinstance(selected, dict) else [])
    if source == "elasticnet":
        selected = layer_result.get("elasticnet_selected_base_vectors", {})
        return _extract_scalar_list(selected.get(selection_key, []) if isinstance(selected, dict) else [])
    raise ValueError(f"Unsupported selection source: {source}")


def _load_layer_selections(
    target_assets: Dict[str, Dict[str, str]],
    layer_idx: int,
) -> Dict[str, Dict[str, Any]]:
    comparison_to_targets: Dict[str, Dict[str, Any]] = {
        comparison: {"selected_by_target": {}, "union_indices": []}
        for comparison in COMPARISON_SPECS
    }

    for target_name, assets in sorted(target_assets.items()):
        path = os.path.join(
            assets["crosslayer_dir"],
            f"{target_name}_train_layer{layer_idx}_evolution.json",
        )
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing crosslayer JSON for {target_name} layer {layer_idx}: {path}")
        data = _read_json(path)
        layer_result = _get_layer_result(data, layer_idx)
        for comparison, (source, selection_key) in COMPARISON_SPECS.items():
            selected_indices = sorted(set(_resolve_selected_indices(layer_result, source, selection_key)))
            comparison_to_targets[comparison]["selected_by_target"][target_name] = selected_indices

    for comparison, payload in comparison_to_targets.items():
        union_indices = sorted(
            {
                idx
                for indices in payload["selected_by_target"].values()
                for idx in indices
            }
        )
        payload["union_indices"] = union_indices
        payload["union_count"] = len(union_indices)

    return comparison_to_targets


def _build_design_matrix(
    rows: List[Dict[str, Any]],
    selected_indices_by_target: Dict[str, List[int]],
    union_indices: List[int],
) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, List[float]]]:
    index_to_col = {int(idx): col_idx for col_idx, idx in enumerate(union_indices)}
    matrix = np.zeros((len(rows), len(union_indices)), dtype=np.float32)

    for row_idx, row in enumerate(rows):
        allowed = set(int(idx) for idx in selected_indices_by_target.get(str(row["target"]), []))
        if not allowed:
            continue
        for latent_idx, value in zip(row["pooled_latent_indices"], row["pooled_latent_acts"]):
            latent_idx = int(latent_idx)
            if latent_idx not in allowed:
                continue
            col_idx = index_to_col.get(latent_idx)
            if col_idx is not None:
                matrix[row_idx, col_idx] = float(value)

    if matrix.shape[1] == 0:
        return matrix, [], {"means": [], "stds": []}

    means = matrix.mean(axis=0, dtype=np.float64)
    stds = matrix.std(axis=0, dtype=np.float64)
    keep_mask = stds >= 1e-6
    filtered = matrix[:, keep_mask]
    kept_indices = [idx for idx, keep in zip(union_indices, keep_mask.tolist()) if keep]
    filtered_means = means[keep_mask]
    filtered_stds = stds[keep_mask]
    standardized = ((filtered - filtered_means) / filtered_stds).astype(np.float32, copy=False)

    feature_metadata = []
    for idx in kept_indices:
        feature_metadata.append(
            {
                "latent_idx": int(idx),
                "selected_targets": sorted(
                    target_name
                    for target_name, indices in selected_indices_by_target.items()
                    if int(idx) in {int(item) for item in indices}
                ),
            }
        )

    standardization = {
        "means": [float(value) for value in filtered_means.tolist()],
        "stds": [float(value) for value in filtered_stds.tolist()],
    }
    return standardized, feature_metadata, standardization


def _summarize_posterior_vector(samples: np.ndarray) -> Dict[str, float]:
    return {
        "posterior_mean": float(np.mean(samples)),
        "posterior_sd": float(np.std(samples, ddof=0)),
        "hdi_3": float(np.quantile(samples, 0.03)),
        "hdi_97": float(np.quantile(samples, 0.97)),
    }


def _extract_elpd_summary(result: Any) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    for attr in ("elpd_waic", "p_waic", "elpd_loo", "p_loo", "se", "warning"):
        if hasattr(result, attr):
            value = getattr(result, attr)
            if isinstance(value, (np.generic, np.ndarray)):
                value = np.asarray(value).item()
            if isinstance(value, (bool, np.bool_)):
                summary[attr] = bool(value)
            elif isinstance(value, (int, float, np.integer, np.floating)):
                summary[attr] = float(value)
            else:
                summary[attr] = value
    return summary


def _compute_model_selection_metrics(pm: Any, idata: Any) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    for metric_name, fn_name in (("waic", "waic"), ("loo", "loo")):
        try:
            metric_result = getattr(pm, fn_name)(idata)
            metrics[metric_name] = {
                "status": "ok",
                **_extract_elpd_summary(metric_result),
            }
        except Exception as exc:
            metrics[metric_name] = {
                "status": "error",
                "reason": str(exc),
            }
    return metrics


def _fit_layer_model(
    rows: List[Dict[str, Any]],
    feature_matrix: np.ndarray,
    feature_metadata: List[Dict[str, Any]],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    try:
        import pymc as pm
    except Exception as exc:
        raise RuntimeError(
            "pymc is required to fit the Bayesian hierarchical logistic model. Install pymc before running this script."
        ) from exc

    target_names = sorted({str(row["target"]) for row in rows})
    pair_keys = sorted({str(row["pair_key"]) for row in rows})
    target_to_idx = {name: idx for idx, name in enumerate(target_names)}
    pair_to_idx = {name: idx for idx, name in enumerate(pair_keys)}

    y = np.asarray([int(row["label"]) for row in rows], dtype=np.int8)
    target_idx = np.asarray([target_to_idx[str(row["target"])] for row in rows], dtype=np.int32)
    pair_idx = np.asarray([pair_to_idx[str(row["pair_key"])] for row in rows], dtype=np.int32)

    coords = {
        "observation": np.arange(len(rows), dtype=np.int32),
        "feature": [f"latent_{item['latent_idx']}" for item in feature_metadata],
        "target": target_names,
        "pair": pair_keys,
    }

    with pm.Model(coords=coords) as model:
        x_data = pm.Data("X", feature_matrix, dims=("observation", "feature"))
        y_data = pm.Data("y", y, dims="observation")
        target_idx_data = pm.Data("target_idx", target_idx, dims="observation")
        pair_idx_data = pm.Data("pair_idx", pair_idx, dims="observation")

        alpha = pm.Normal("alpha", mu=0.0, sigma=1.5)
        sigma_target = pm.HalfNormal("sigma_target", sigma=1.0)
        target_offset = pm.Normal("target_offset", mu=0.0, sigma=1.0, dims="target")
        target_effect = pm.Deterministic("target_effect", target_offset * sigma_target, dims="target")
        sigma_pair = pm.HalfNormal("sigma_pair", sigma=1.0)
        pair_offset = pm.Normal("pair_offset", mu=0.0, sigma=1.0, dims="pair")
        pair_effect = pm.Deterministic("pair_effect", pair_offset * sigma_pair, dims="pair")
        sigma_beta = pm.HalfNormal("sigma_beta", sigma=1.0)
        beta = pm.Normal("beta", mu=0.0, sigma=sigma_beta, dims="feature")

        logits = alpha + target_effect[target_idx_data] + pair_effect[pair_idx_data] + pm.math.dot(x_data, beta)
        pm.Bernoulli("obs", logit_p=logits, observed=y_data, dims="observation")

        idata = pm.sample(
            draws=args.draws,
            tune=args.tune,
            chains=args.chains,
            cores=args.cores,
            target_accept=args.target_accept,
            random_seed=args.random_seed,
            return_inferencedata=True,
            idata_kwargs={"log_likelihood": True},
        )

    beta_samples = np.asarray(idata.posterior["beta"]).reshape(-1, len(feature_metadata))
    target_samples = np.asarray(idata.posterior["target_effect"]).reshape(-1, len(target_names))
    alpha_samples = np.asarray(idata.posterior["alpha"]).reshape(-1)
    sigma_beta_samples = np.asarray(idata.posterior["sigma_beta"]).reshape(-1)
    sigma_target_samples = np.asarray(idata.posterior["sigma_target"]).reshape(-1)
    sigma_pair_samples = np.asarray(idata.posterior["sigma_pair"]).reshape(-1)

    feature_summaries = []
    for col_idx, feature in enumerate(feature_metadata):
        feature_summaries.append({**feature, **_summarize_posterior_vector(beta_samples[:, col_idx])})

    target_summaries = []
    for col_idx, target_name in enumerate(target_names):
        target_summaries.append({"target": target_name, **_summarize_posterior_vector(target_samples[:, col_idx])})

    return {
        "idata": idata,
        "feature_summaries": feature_summaries,
        "target_summaries": target_summaries,
        "model_selection": _compute_model_selection_metrics(pm, idata),
        "scalar_summaries": {
            "alpha": _summarize_posterior_vector(alpha_samples),
            "sigma_beta": _summarize_posterior_vector(sigma_beta_samples),
            "sigma_target": _summarize_posterior_vector(sigma_target_samples),
            "sigma_pair": _summarize_posterior_vector(sigma_pair_samples),
        },
        "num_observations": int(len(rows)),
        "num_features": int(len(feature_metadata)),
        "num_targets": int(len(target_names)),
        "num_pairs": int(len(pair_keys)),
    }


def _load_layer_rows(activations_path: str) -> List[Dict[str, Any]]:
    df = pd.read_parquet(activations_path)
    return df.to_dict(orient="records")


def _save_comparison_outputs(
    layer_idx: int,
    comparison_name: str,
    selection_payload: Dict[str, Any],
    feature_metadata: List[Dict[str, Any]],
    standardization: Dict[str, List[float]],
    model_results: Dict[str, Any],
    output_dir: str,
) -> Dict[str, str]:
    base_dir = os.path.join(output_dir, comparison_name)
    posterior_dir = os.path.join(base_dir, "posterior")
    summary_dir = os.path.join(base_dir, "summaries")
    os.makedirs(posterior_dir, exist_ok=True)
    os.makedirs(summary_dir, exist_ok=True)

    feature_summary_path = os.path.join(summary_dir, f"layer{layer_idx}_feature_coefficients.parquet")
    pd.DataFrame(model_results["feature_summaries"]).to_parquet(feature_summary_path, index=False)

    target_summary_path = os.path.join(summary_dir, f"layer{layer_idx}_target_effects.parquet")
    pd.DataFrame(model_results["target_summaries"]).to_parquet(target_summary_path, index=False)

    metadata_path = os.path.join(summary_dir, f"layer{layer_idx}_metadata.json")
    metadata = {
        "layer": int(layer_idx),
        "comparison": comparison_name,
        "selection_payload": selection_payload,
        "selected_features": feature_metadata,
        "standardization": standardization,
        "model_selection": model_results["model_selection"],
        "scalar_summaries": model_results["scalar_summaries"],
        "num_observations": int(model_results["num_observations"]),
        "num_features": int(model_results["num_features"]),
        "num_targets": int(model_results["num_targets"]),
        "num_pairs": int(model_results["num_pairs"]),
    }
    save_json_results(metadata, metadata_path)

    posterior_path = os.path.join(posterior_dir, f"layer{layer_idx}_posterior.nc")
    model_results["idata"].to_netcdf(posterior_path)

    return {
        "feature_summary_path": feature_summary_path,
        "target_summary_path": target_summary_path,
        "metadata_path": metadata_path,
        "posterior_path": posterior_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit PyMC Bayesian hierarchical logistic models from precomputed dev activation parquet files using only the pre-selected top3/top100/lasso/elasticnet indices for each layer."
        )
    )
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--activations-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--targets", default=None)
    parser.add_argument("--exclude-targets", default="know")
    parser.add_argument("--comparisons", default=None, help="Optional comma-separated subset of comparisons.")
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--tune", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=2)
    parser.add_argument("--cores", type=int, default=1)
    parser.add_argument("--target-accept", type=float, default=0.9)
    parser.add_argument("--random-seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or os.path.join(args.input_dir, "bayesian_dev_analysis")
    activations_dir = args.activations_dir or os.path.join(output_dir, "activations")
    os.makedirs(output_dir, exist_ok=True)

    include_targets = _parse_csv_list(args.targets)
    exclude_targets = _parse_csv_list(args.exclude_targets) or sorted(DEFAULT_EXCLUDED_TARGETS)
    target_assets = _discover_target_directories(args.input_dir, include_targets, exclude_targets)
    activation_files = _discover_activation_files(activations_dir)
    selected_comparisons = _parse_csv_list(args.comparisons) or list(COMPARISON_SPECS.keys())
    unknown = sorted(set(selected_comparisons) - set(COMPARISON_SPECS.keys()))
    if unknown:
        raise ValueError(f"Unknown comparisons: {unknown}")

    run_summary: Dict[str, Any] = {
        "input_dir": args.input_dir,
        "activations_dir": activations_dir,
        "targets": sorted(target_assets.keys()),
        "excluded_targets": sorted(exclude_targets),
        "comparisons": selected_comparisons,
        "layers": sorted(activation_files.keys()),
        "layer_outputs": {},
    }

    for layer_idx in sorted(activation_files.keys()):
        print(f"[bayes] loading activations for layer {layer_idx}")
        layer_rows = _load_layer_rows(activation_files[layer_idx])
        layer_selections = _load_layer_selections(target_assets, layer_idx)
        layer_summary: Dict[str, Any] = {
            "activations_path": activation_files[layer_idx],
            "comparisons": {},
            "model_selection_ranking": [],
        }

        for comparison_name in selected_comparisons:
            payload = layer_selections[comparison_name]
            union_indices = payload["union_indices"]
            print(
                f"[bayes] layer {layer_idx} comparison {comparison_name}: using {len(union_indices)} selected indices"
            )
            feature_matrix, feature_metadata, standardization = _build_design_matrix(
                layer_rows,
                payload["selected_by_target"],
                union_indices,
            )
            if feature_matrix.shape[1] == 0:
                layer_summary["comparisons"][comparison_name] = {
                    "status": "skipped",
                    "reason": "no_variable_selected_indices",
                    "union_count": len(union_indices),
                }
                continue

            model_results = _fit_layer_model(layer_rows, feature_matrix, feature_metadata, args)
            saved_paths = _save_comparison_outputs(
                layer_idx,
                comparison_name,
                payload,
                feature_metadata,
                standardization,
                model_results,
                output_dir,
            )
            layer_summary["comparisons"][comparison_name] = {
                "status": "fit",
                "union_count": len(union_indices),
                "num_selected_features": len(feature_metadata),
                "model_selection": model_results["model_selection"],
                **saved_paths,
            }

        ranking_rows = []
        for comparison_name, comparison_summary in layer_summary["comparisons"].items():
            if comparison_summary.get("status") != "fit":
                continue
            metrics = comparison_summary.get("model_selection", {})
            waic = metrics.get("waic", {})
            loo = metrics.get("loo", {})
            ranking_rows.append(
                {
                    "comparison": comparison_name,
                    "waic_elpd": waic.get("elpd_waic"),
                    "waic_p": waic.get("p_waic"),
                    "loo_elpd": loo.get("elpd_loo"),
                    "loo_p": loo.get("p_loo"),
                }
            )
        ranking_rows.sort(
            key=lambda row: (
                -(row["loo_elpd"] if row["loo_elpd"] is not None else float("-inf")),
                -(row["waic_elpd"] if row["waic_elpd"] is not None else float("-inf")),
                row["comparison"],
            )
        )
        layer_summary["model_selection_ranking"] = ranking_rows
        if ranking_rows:
            ranking_path = os.path.join(output_dir, f"layer{layer_idx}_model_selection.json")
            save_json_results(ranking_rows, ranking_path)
            layer_summary["model_selection_ranking_path"] = ranking_path

        run_summary["layer_outputs"][str(layer_idx)] = layer_summary

    summary_path = os.path.join(output_dir, "run_summary.json")
    save_json_results(run_summary, summary_path)
    print(f"Saved run summary: {summary_path}")


if __name__ == "__main__":
    main()
