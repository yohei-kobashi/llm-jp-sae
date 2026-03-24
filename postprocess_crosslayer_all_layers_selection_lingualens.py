#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import re
from glob import glob
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.linear_model import ElasticNetCV, LassoCV

from analyzer_lingualens import TrainSaeLinguisticAnalyzer
from LinguaLens.lingualens.utils import load_text_data, save_json_results, validate_layer_indices
from model import normalize_activation

PER_LAYER_TOP_COUNTS = (3, 10)


def _collect_layer_json_paths(input_dir: str) -> List[str]:
    paths = sorted(glob(os.path.join(input_dir, '*_layer*_evolution.json')))
    if not paths:
        raise FileNotFoundError(
            f'No per-layer crosslayer JSON files found in directory: {input_dir}'
        )
    return paths


def _infer_feature_name(paths: List[str]) -> str:
    feature_names = set()
    for path in paths:
        name = os.path.basename(path)
        match = re.match(r'(.+)_layer\d+_evolution\.json$', name)
        if not match:
            raise ValueError(f'Unexpected per-layer filename format: {path}')
        feature_names.add(match.group(1))
    if len(feature_names) != 1:
        raise ValueError(f'Expected exactly one feature prefix, got: {sorted(feature_names)}')
    return next(iter(feature_names))


def _load_layer_results(input_dir: str) -> List[Tuple[int, Dict[str, Any]]]:
    loaded: List[Tuple[int, Dict[str, Any]]] = []
    for path in _collect_layer_json_paths(input_dir):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        layer_idx = data.get('layer_idx')
        if layer_idx is None:
            match = re.search(r'_layer(\d+)_evolution\.json$', os.path.basename(path))
            if not match:
                raise ValueError(f'Could not infer layer index from: {path}')
            layer_idx = int(match.group(1))
        loaded.append((int(layer_idx), data))
    loaded.sort(key=lambda item: item[0])
    return loaded


def _get_single_layer_result(data: Dict[str, Any], layer: int) -> Dict[str, Any]:
    base_results = data.get('base_results', {})
    layer_results = base_results.get('layer_results', {})
    return layer_results.get(layer, layer_results.get(str(layer), {}))


def _get_full_stats(layer_result: Dict[str, Any]) -> Dict[str, Any]:
    full_stats = layer_result.get('full_stats', {})
    return full_stats if isinstance(full_stats, dict) else {}


def _build_candidate_sets(
    layer_entries: List[Tuple[int, Dict[str, Any]]],
    top_counts: List[int],
) -> Tuple[str, List[int], Dict[int, Dict[str, Any]], Dict[str, Any]]:
    feature_file: Optional[str] = None
    layers = validate_layer_indices([layer for layer, _ in layer_entries])
    top_count_candidates: Dict[int, Dict[str, Any]] = {}
    source_summary: Dict[str, Any] = {
        'layers': layers,
        'per_layer_top_counts': top_counts,
        'layer_feature_summaries': {},
    }

    for top_count in top_counts:
        candidate_features: List[Dict[str, Any]] = []
        selected_by_layer: Dict[int, List[int]] = {}

        for layer, data in layer_entries:
            base_results = data.get('base_results', {})
            if feature_file is None:
                feature_file = base_results.get('feature_file')

            layer_result = _get_single_layer_result(data, layer)
            full_stats = _get_full_stats(layer_result)
            top_features = layer_result.get('top_100_features') or layer_result.get('top_features') or []
            if len(top_features) < top_count:
                raise ValueError(
                    f'Layer {layer} has only {len(top_features)} top features, cannot build top_{top_count}.'
                )

            selected_base_vectors: List[int] = []
            for rank_in_layer, feature_pair in enumerate(top_features[:top_count], start=1):
                if not isinstance(feature_pair, (list, tuple)) or len(feature_pair) < 2:
                    raise ValueError(f'Invalid feature pair in layer {layer}: {feature_pair!r}')
                base_vector = int(feature_pair[0])
                frc = float(feature_pair[1])
                stats = full_stats.get(base_vector, full_stats.get(str(base_vector), {}))
                feature_id = f'layer{layer}_bv{base_vector}'
                candidate_features.append(
                    {
                        'feature_id': feature_id,
                        'layer': int(layer),
                        'base_vector': base_vector,
                        'rank_within_layer': int(rank_in_layer),
                        'frc': frc,
                        'ps': float(stats.get('ps', 0.0)),
                        'pn': float(stats.get('pn', 0.0)),
                        'avg_max_activation': float(stats.get('avg_max_activation', 0.0)),
                    }
                )
                selected_base_vectors.append(base_vector)

            selected_by_layer[int(layer)] = selected_base_vectors
            source_summary['layer_feature_summaries'][str(layer)] = {
                'available_top_features': len(top_features),
                'source': 'top_100_features' if layer_result.get('top_100_features') else 'top_features',
            }

        top_count_candidates[top_count] = {
            'candidate_features': candidate_features,
            'selected_by_layer': selected_by_layer,
        }

    if feature_file is None:
        raise ValueError('Could not infer feature_file from per-layer crosslayer outputs.')

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
        runtimes = {int(layer): self.analyzer._get_sae_model(int(layer)) for layer in layers}

        sentence_feature_rows: List[Dict[str, Any]] = []
        total_batches = max(1, (len(lines) + self.batch_size - 1) // self.batch_size)
        log_interval = max(1, total_batches // 10)

        for batch_idx, start in enumerate(range(0, len(lines), self.batch_size), start=1):
            if batch_idx == 1 or batch_idx == total_batches or batch_idx % log_interval == 0:
                print(
                    f'[all-layers] batch {batch_idx}/{total_batches} '
                    f'(examples {start + 1}-{min(start + self.batch_size, len(lines))}/{len(lines)})'
                )

            batch_lines = lines[start : start + self.batch_size]
            enc = self.analyzer.tokenizer(
                batch_lines,
                return_tensors='pt',
                padding=True,
                truncation=True,
            )
            input_ids = enc['input_ids'].to(self.analyzer.device)
            attention_mask = enc.get('attention_mask')
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids)
            attention_mask = attention_mask.to(self.analyzer.device)

            batch_rows = []
            for row_idx in range(input_ids.size(0)):
                sentence_id = start + row_idx + 1
                batch_rows.append(
                    {
                        'sentence_id': int(sentence_id),
                        'label': 1 if sentence_id % 2 == 1 else 0,
                        'line_type': 'original' if sentence_id % 2 == 1 else 'minimal_pair',
                        'feature_activations': {},
                    }
                )

            with torch.no_grad():
                _ = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

            token_mask = attention_mask[:, 1:]
            for layer in layers:
                runtime = runtimes[int(layer)]
                sae = runtime['sae']
                hook = runtime['hook']
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
                        f'layer{layer}_bv{base_vector}': 0.0 for base_vector in layer_candidates
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
                                feature_id = f'layer{layer}_bv{int(base_vector)}'
                                current_value = layer_activation_map[feature_id]
                                if float(value) > current_value:
                                    layer_activation_map[feature_id] = float(value)

                    batch_rows[row_idx]['feature_activations'].update(layer_activation_map)

                hook.output = None

            sentence_feature_rows.extend(batch_rows)

        return sentence_feature_rows

    def clear_cache(self) -> None:
        self.analyzer.clear_cache()


def _standardize_matrix(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    feature_means = matrix.mean(axis=0, dtype=np.float64)
    feature_stds = matrix.std(axis=0, dtype=np.float64)
    feature_stds[feature_stds < 1e-6] = 1.0
    standardized = (matrix - feature_means) / feature_stds
    return standardized.astype(np.float64, copy=False), feature_means, feature_stds


def _build_feature_matrix(
    sentence_feature_rows: List[Dict[str, Any]],
    candidate_features: List[Dict[str, Any]],
) -> Tuple[np.ndarray, np.ndarray]:
    feature_ids = [feature['feature_id'] for feature in candidate_features]
    labels = np.asarray([float(row.get('label', 0)) for row in sentence_feature_rows], dtype=np.float32)
    matrix = np.zeros((len(sentence_feature_rows), len(feature_ids)), dtype=np.float32)

    for row_idx, row in enumerate(sentence_feature_rows):
        feature_acts = row.get('feature_activations', {})
        for col_idx, feature_id in enumerate(feature_ids):
            matrix[row_idx, col_idx] = float(feature_acts.get(feature_id, 0.0))

    return matrix, labels


def _ranked_feature_weights(
    ranked_indices: List[int],
    candidate_features: List[Dict[str, Any]],
    coefficients: np.ndarray,
) -> List[Dict[str, Any]]:
    ranked: List[Dict[str, Any]] = []
    for idx in ranked_indices:
        feature = candidate_features[idx]
        ranked.append(
            {
                'feature_id': feature['feature_id'],
                'layer': int(feature['layer']),
                'base_vector': int(feature['base_vector']),
                'rank_within_layer': int(feature['rank_within_layer']),
                'weight': float(coefficients[idx]),
                'abs_weight': float(abs(coefficients[idx])),
                'frc': float(feature['frc']),
                'ps': float(feature['ps']),
                'pn': float(feature['pn']),
                'avg_max_activation': float(feature['avg_max_activation']),
            }
        )
    return ranked


def _fit_regularized_selector(
    sentence_feature_rows: List[Dict[str, Any]],
    candidate_features: List[Dict[str, Any]],
    method: str,
) -> Dict[str, Any]:
    if method not in {'lasso', 'elasticnet'}:
        raise ValueError(f'Unsupported method: {method}')

    feature_matrix, labels = _build_feature_matrix(sentence_feature_rows, candidate_features)
    if feature_matrix.shape[1] == 0:
        return {'status': 'skipped', 'reason': 'no_candidate_features', 'method': method}
    if len(sentence_feature_rows) < 2 or len(np.unique(labels)) < 2:
        return {'status': 'skipped', 'reason': 'insufficient_labels', 'method': method}

    standardized_matrix, means, stds = _standardize_matrix(feature_matrix)
    cv_folds = max(2, min(5, len(sentence_feature_rows)))

    if method == 'lasso':
        model = LassoCV(cv=cv_folds, random_state=42, max_iter=10000, precompute=False)
        model.fit(standardized_matrix, labels)
        coefficients = model.coef_.astype(np.float32, copy=False)
        metadata: Dict[str, Any] = {
            'alpha': float(model.alpha_),
            'intercept': float(model.intercept_),
            'score_r2': float(model.score(standardized_matrix, labels)),
        }
    else:
        model = ElasticNetCV(
            l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99],
            cv=cv_folds,
            random_state=42,
            max_iter=10000,
        )
        model.fit(standardized_matrix, labels)
        coefficients = model.coef_.astype(np.float32, copy=False)
        metadata = {
            'alpha': float(model.alpha_),
            'l1_ratio': float(model.l1_ratio_),
            'intercept': float(model.intercept_),
            'score_r2': float(model.score(standardized_matrix, labels)),
        }

    ranked_indices = sorted(
        [idx for idx in range(len(candidate_features)) if abs(coefficients[idx]) > 1e-8],
        key=lambda idx: (-abs(coefficients[idx]), candidate_features[idx]['layer'], candidate_features[idx]['base_vector']),
    )

    return {
        'status': 'fit',
        'method': method,
        'num_examples': int(len(sentence_feature_rows)),
        'num_candidate_features': int(len(candidate_features)),
        'non_zero_feature_count': int(len(ranked_indices)),
        'feature_weights': _ranked_feature_weights(ranked_indices, candidate_features, coefficients),
        'selected_features': [
            {
                'feature_id': candidate_features[idx]['feature_id'],
                'layer': int(candidate_features[idx]['layer']),
                'base_vector': int(candidate_features[idx]['base_vector']),
            }
            for idx in ranked_indices
        ],
        'standardization': {
            'means': [float(value) for value in means.tolist()],
            'stds': [float(value) for value in stds.tolist()],
        },
        **metadata,
    }


def _candidate_summary(candidate_payload: Dict[str, Any]) -> Dict[str, Any]:
    candidate_features = candidate_payload['candidate_features']
    selected_by_layer = candidate_payload['selected_by_layer']
    return {
        'status': 'ready',
        'num_candidate_features': int(len(candidate_features)),
        'candidate_features': candidate_features,
        'selected_by_layer': {str(layer): values for layer, values in selected_by_layer.items()},
    }


def _parse_torch_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        'bfloat16': torch.bfloat16,
        'float16': torch.float16,
        'float32': torch.float32,
    }
    if dtype_name not in mapping:
        raise ValueError(f'Unsupported torch dtype: {dtype_name}')
    return mapping[dtype_name]


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Select cross-layer features from all-layer activations and output 6 selection patterns.'
    )
    parser.add_argument('--input-dir', required=True, help='Directory containing per-layer crosslayer JSON files for one target.')
    parser.add_argument('--model-path', required=True, help='Base model path or Hugging Face id.')
    parser.add_argument('--sae-path-template', required=True, help='Template path to sae_layer{}.pth')
    parser.add_argument('--output-dir', default=None, help='Output directory. Default: same as --input-dir.')
    parser.add_argument('--device', default=None)
    parser.add_argument('--k', type=int, default=32)
    parser.add_argument('--normalization', default='Scalar')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--torch-dtype', default='bfloat16')
    args = parser.parse_args()

    output_dir = args.output_dir or args.input_dir
    os.makedirs(output_dir, exist_ok=True)

    layer_entries = _load_layer_results(args.input_dir)
    feature_name = _infer_feature_name(_collect_layer_json_paths(args.input_dir))
    feature_file, layers, top_count_candidates, source_summary = _build_candidate_sets(
        layer_entries,
        list(PER_LAYER_TOP_COUNTS),
    )

    collector = TrainSaeAllLayersAnalyzer(
        model_path=args.model_path,
        sae_path_template=args.sae_path_template,
        device=args.device,
        k=args.k,
        normalization=args.normalization,
        batch_size=args.batch_size,
        torch_dtype=_parse_torch_dtype(args.torch_dtype),
    )

    try:
        sentence_feature_rows = collector.collect_sentence_feature_rows(
            feature_file,
            layers,
            top_count_candidates[10]['selected_by_layer'],
        )
    finally:
        collector.clear_cache()

    top3_candidates = top_count_candidates[3]
    top10_candidates = top_count_candidates[10]

    results: Dict[str, Any] = {
        'feature_name': feature_name,
        'feature_file': feature_file,
        'layers': layers,
        'source_summary': source_summary,
        'num_examples': int(len(sentence_feature_rows)),
        'patterns': {
            'all_top3': _candidate_summary(top3_candidates),
            'all_top10': _candidate_summary(top10_candidates),
            'lasso_from_top3': _fit_regularized_selector(
                sentence_feature_rows,
                top3_candidates['candidate_features'],
                'lasso',
            ),
            'elasticnet_from_top3': _fit_regularized_selector(
                sentence_feature_rows,
                top3_candidates['candidate_features'],
                'elasticnet',
            ),
            'lasso_from_top10': _fit_regularized_selector(
                sentence_feature_rows,
                top10_candidates['candidate_features'],
                'lasso',
            ),
            'elasticnet_from_top10': _fit_regularized_selector(
                sentence_feature_rows,
                top10_candidates['candidate_features'],
                'elasticnet',
            ),
        },
    }

    output_path = os.path.join(output_dir, f'{feature_name}_all_layers_selection_summary.json')
    save_json_results(results, output_path)
    print(f'Saved all-layer selection summary: {output_path}')


if __name__ == '__main__':
    main()
