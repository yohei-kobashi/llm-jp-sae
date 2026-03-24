#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
from glob import glob
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
import torch

from analyzer_lingualens import TrainSaeLinguisticAnalyzer
from LinguaLens.lingualens.utils import load_text_data, save_json_results
from model import normalize_activation



def _parse_torch_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")
    return mapping[dtype_name]


def _parse_csv_list(raw_value: Optional[str]) -> List[str]:
    if not raw_value:
        return []
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def _discover_target_directories(
    input_dir: str,
    include_targets: Optional[List[str]],
    split: str,
) -> Dict[str, Dict[str, str]]:
    include_set = set(include_targets or [])
    discovered: Dict[str, Dict[str, str]] = {}

    suffix = f"_{split}.txt"
    for split_path in sorted(glob(os.path.join(input_dir, "*", "data", f"*{suffix}"))):
        target_name = os.path.basename(split_path)[: -len(suffix)]
        if include_set and target_name not in include_set:
            continue
        discovered[target_name] = {"split_path": split_path}

    if include_set:
        missing_targets = sorted(include_set - set(discovered.keys()))
        if missing_targets:
            raise FileNotFoundError(
                f"Could not find target directories under {input_dir}: {missing_targets}"
            )

    if not discovered:
        raise FileNotFoundError(
            f"No eligible target/data/*_{split}.txt files found under {input_dir}."
        )

    return discovered


class DevActivationCollector:
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
        self.batch_size = int(batch_size)
        self.normalization = normalization

    def infer_all_layers(self) -> List[int]:
        layer_modules = self.analyzer._resolve_layer_modules()
        return list(range(0, len(layer_modules["layers"]) + 1))

    def collect_layer_rows(
        self,
        layer_idx: int,
        target_assets: Dict[str, Dict[str, str]],
    ) -> List[Dict[str, Any]]:
        runtime = self.analyzer._get_sae_model(layer_idx)
        model = self.analyzer._load_base_model()
        sae = runtime["sae"]
        hook = runtime["hook"]
        collected_rows: List[Dict[str, Any]] = []

        for target_name, assets in sorted(target_assets.items()):
            lines = load_text_data(assets["split_path"])
            total_batches = max(1, (len(lines) + self.batch_size - 1) // self.batch_size)
            log_interval = max(1, total_batches // 10)

            for batch_idx, start in enumerate(range(0, len(lines), self.batch_size), start=1):
                if batch_idx == 1 or batch_idx == total_batches or batch_idx % log_interval == 0:
                    print(
                        f"[layer {layer_idx}] {target_name}: batch {batch_idx}/{total_batches} "
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

                with torch.no_grad():
                    _ = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)

                acts = hook.output
                if isinstance(acts, tuple):
                    acts = acts[0]
                if acts.dim() == 2:
                    acts = acts.unsqueeze(0)
                acts = acts[:, 1:, :]
                token_mask = attention_mask[:, 1:]

                for row_idx in range(input_ids.size(0)):
                    valid_len = int(token_mask[row_idx].sum().item())
                    pooled_latent_acts: Dict[int, float] = {}
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
                                value = float(value)
                                if value <= 0.0:
                                    continue
                                base_vector = int(base_vector)
                                current = pooled_latent_acts.get(base_vector, 0.0)
                                if value > current:
                                    pooled_latent_acts[base_vector] = value

                    line_idx = start + row_idx
                    line_number = line_idx + 1
                    pair_id = line_idx // 2
                    sorted_latents = sorted(pooled_latent_acts.items())
                    collected_rows.append(
                        {
                            "target": target_name,
                            "source_path": assets["split_path"],
                            "layer": int(layer_idx),
                            "line_index": int(line_idx),
                            "line_number": int(line_number),
                            "pair_id": int(pair_id),
                            "pair_key": f"{target_name}::{pair_id}",
                            "label": 1 if line_number % 2 == 1 else 0,
                            "line_type": "original" if line_number % 2 == 1 else "minimal_pair",
                            "text": batch_lines[row_idx],
                            "pooled_latent_indices": [latent for latent, _ in sorted_latents],
                            "pooled_latent_acts": [act for _, act in sorted_latents],
                            "active_latent_count": len(sorted_latents),
                        }
                    )
                hook.output = None

        return collected_rows

    def clear_cache(self) -> None:
        self.analyzer.clear_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect and save dev-set SAE activations for all layers without fitting PyMC."
    )
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--sae-path-template", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--targets", default=None)
    parser.add_argument(
        "--split",
        choices=("train", "dev", "test"),
        default="dev",
        help="Dataset split to collect activations from. Default: dev",
    )
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--normalization", type=str, default="Scalar")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--torch-dtype", type=str, default="bfloat16")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.input_dir
    activations_dir = os.path.join(output_dir, "activations")
    os.makedirs(activations_dir, exist_ok=True)

    include_targets = _parse_csv_list(args.targets)
    target_assets = _discover_target_directories(
        args.input_dir,
        include_targets,
        args.split,
    )

    collector = DevActivationCollector(
        model_path=args.model_path,
        sae_path_template=args.sae_path_template,
        device=args.device,
        k=args.k,
        normalization=args.normalization,
        batch_size=args.batch_size,
        torch_dtype=_parse_torch_dtype(args.torch_dtype),
    )

    try:
        layers = collector.infer_all_layers()
        run_summary: Dict[str, Any] = {
            "model_path": args.model_path,
            "input_dir": args.input_dir,
            "sae_path_template": args.sae_path_template,
            "targets": sorted(target_assets.keys()),
            "split": args.split,
            "layers": layers,
            "layer_outputs": {},
        }

        for layer_idx in layers:
            print(f"[collect] collecting activations for layer {layer_idx}")
            layer_rows = collector.collect_layer_rows(layer_idx, target_assets)
            output_path = os.path.join(
                activations_dir,
                f"layer{layer_idx}_{args.split}_activations.parquet",
            )
            pd.DataFrame(layer_rows).to_parquet(output_path, index=False)
            run_summary["layer_outputs"][str(layer_idx)] = {
                "num_rows": len(layer_rows),
                "activations_path": output_path,
            }
            print(f"Saved activations: {output_path}")

        summary_path = os.path.join(output_dir, "activation_collection_summary.json")
        save_json_results(run_summary, summary_path)
        print(f"Saved activation collection summary: {summary_path}")
    finally:
        collector.clear_cache()


if __name__ == "__main__":
    main()
