"""
LinguisticAnalyzer extension for Llama Scope sparse dictionaries.

This module mirrors analyzer_lingualens.py but replaces the local train.py SAE
loader with Language-Model-SAEs / Llama Scope loading based on
SparseDictionary.from_pretrained and TransformerLens hook caches.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict, List, Optional

import torch

from analyzer_lingualens import BaseTrainSaeLinguisticAnalyzer, _parse_layers, _parse_torch_dtype
from LinguaLens.lingualens.metrics import compute_layer_stats, get_top_features_by_frc
from LinguaLens.lingualens.utils import ProgressLogger, save_json_results

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
LM_SAES_SRC = os.path.join(PROJECT_ROOT, "Language-Model-SAEs", "src")
if LM_SAES_SRC not in sys.path:
    sys.path.insert(0, LM_SAES_SRC)

from lm_saes.models.sparse_dictionary import SparseDictionary
from transformer_lens import HookedTransformer


class LlamaScopeLinguisticAnalyzer(BaseTrainSaeLinguisticAnalyzer):
    """
    LinguisticAnalyzer compatible with Llama Scope sparse dictionaries.

    `sae_path_template` should be a format string that resolves a layer-specific
    local path or HuggingFace identifier accepted by SparseDictionary, such as:
        OpenMOSS-Team/Llama-Scope-2-Qwen3-1.7B:transcoder/8x/k128/layer{}_transcoder_8x_k128
    """

    def __init__(
        self,
        model_path: str,
        sae_path_template: str,
        device: Optional[str] = None,
        batch_size: int = 8,
        torch_dtype: torch.dtype = torch.bfloat16,
        prepend_bos: bool = True,
        fold_activation_scale: bool = False,
    ):
        super().__init__(
            model_path=model_path,
            sae_path_template=sae_path_template,
            device=device,
        )
        self.batch_size = int(batch_size)
        self.torch_dtype = torch_dtype
        self.prepend_bos = bool(prepend_bos)
        self.fold_activation_scale = bool(fold_activation_scale)
        self._base_model: Optional[HookedTransformer] = None

    def _load_base_model(self) -> HookedTransformer:
        if self._base_model is not None:
            return self._base_model

        model = HookedTransformer.from_pretrained(self.model_path)
        model = model.to(self.device)
        model.eval()
        self._base_model = model
        return model

    def _load_llamascope_sae(self, layer_idx: int) -> SparseDictionary:
        sae_path = self.sae_path_template.format(layer_idx)
        sae = SparseDictionary.from_pretrained(
            sae_path,
            fold_activation_scale=self.fold_activation_scale,
        ).to(self.device)
        sae.eval()
        return sae

    def _get_sae_model(self, layer_idx: int) -> Dict[str, Any]:
        if layer_idx in self._model_cache:
            return self._model_cache[layer_idx]

        sae = self._load_llamascope_sae(layer_idx)
        runtime = {
            "sae": sae,
            "hook_point_in": sae.cfg.hook_point_in,
        }
        self._model_cache[layer_idx] = runtime
        return runtime

    def _prepare_tokens(self, batch_lines: List[str]) -> torch.Tensor:
        enc = self.tokenizer(
            batch_lines,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        tokens = enc["input_ids"]
        bos_token_id = self.tokenizer.bos_token_id
        if (
            self.prepend_bos
            and bos_token_id is not None
            and tokens.size(1) > 0
            and not torch.all(tokens[:, 0] == bos_token_id)
        ):
            bos_column = torch.full(
                (tokens.size(0), 1),
                bos_token_id,
                dtype=tokens.dtype,
            )
            tokens = torch.cat([bos_column, tokens], dim=1)
        return tokens.to(self.device)

    def _build_token_mask(self, tokens: torch.Tensor) -> torch.Tensor:
        mask = torch.ones_like(tokens, dtype=torch.bool)
        pad_token_id = self.tokenizer.pad_token_id
        bos_token_id = self.tokenizer.bos_token_id
        if pad_token_id is not None:
            mask &= tokens.ne(pad_token_id)
        if bos_token_id is not None:
            mask &= tokens.ne(bos_token_id)
        return mask

    def _collect_layer_structured_batch(
        self,
        tokens: torch.Tensor,
        token_mask: torch.Tensor,
        feature_acts: torch.Tensor,
        start: int,
        retain_sentence_activations: bool,
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        structured_batch: List[Dict[str, Any]] = []
        sentence_feature_rows: List[Dict[str, Any]] = []

        if isinstance(feature_acts, tuple):
            feature_acts = feature_acts[0]
        if feature_acts.dim() == 2:
            feature_acts = feature_acts.unsqueeze(0)

        for i in range(tokens.size(0)):
            valid_positions = token_mask[i].nonzero(as_tuple=False).flatten()
            token_ids = tokens[i, valid_positions].detach().cpu().tolist()
            str_tokens = self.tokenizer.convert_ids_to_tokens(token_ids)
            str_tokens = [tok.replace("▁", " ") for tok in str_tokens]

            sentence_tokens: List[Dict[str, Any]] = []
            sentence_latent_activations: Dict[int, float] = {}
            if len(valid_positions) > 0:
                sentence_feature_acts = feature_acts[i, valid_positions, :].detach().cpu()

                for token, latent_vector in zip(str_tokens, sentence_feature_acts):
                    nz = (latent_vector > 0).nonzero(as_tuple=False).flatten()
                    token_activations = []
                    for feature_idx_tensor in nz:
                        feature_idx = int(feature_idx_tensor.item())
                        value = float(latent_vector[feature_idx].item())
                        current_value = sentence_latent_activations.get(feature_idx, 0.0)
                        if value > current_value:
                            sentence_latent_activations[feature_idx] = value
                        token_activations.append(
                            {
                                "base_vector": feature_idx,
                                "activation": value,
                            }
                        )
                    sentence_tokens.append(
                        {"token": token, "activations": token_activations}
                    )

            structured_batch.append(
                {
                    "sentence_id": start + i + 1,
                    "tokens": sentence_tokens,
                }
            )
            if retain_sentence_activations:
                sentence_id = start + i + 1
                sentence_feature_rows.append(
                    {
                        "sentence_id": sentence_id,
                        "label": 1 if sentence_id % 2 == 1 else 0,
                        "line_type": "original" if sentence_id % 2 == 1 else "minimal_pair",
                        "latent_activations": sentence_latent_activations,
                    }
                )

        return structured_batch, sentence_feature_rows

    def _analyze_multiple_layers(
        self,
        lines: List[str],
        layers: List[int],
        retain_sentence_activations: bool = False,
    ) -> Dict[int, Dict[str, Any]]:
        model = self._load_base_model()
        runtimes = {layer_idx: self._get_sae_model(layer_idx) for layer_idx in layers}
        hook_points = [str(runtimes[layer_idx]["hook_point_in"]) for layer_idx in layers]
        structured_data_by_layer = {layer_idx: [] for layer_idx in layers}
        sentence_rows_by_layer = {layer_idx: [] for layer_idx in layers}
        total_batches = max(1, (len(lines) + self.batch_size - 1) // self.batch_size)
        log_interval = max(1, total_batches // 10)

        for batch_idx, start in enumerate(range(0, len(lines), self.batch_size), start=1):
            if batch_idx == 1 or batch_idx == total_batches or batch_idx % log_interval == 0:
                print(
                    f"  shared forward: batch {batch_idx}/{total_batches} "
                    f"(examples {start + 1}-{min(start + self.batch_size, len(lines))}/{len(lines)})"
                )

            batch_lines = lines[start : start + self.batch_size]
            tokens = self._prepare_tokens(batch_lines)
            token_mask = self._build_token_mask(tokens)

            with torch.no_grad():
                _, cache = model.run_with_cache(tokens, names_filter=hook_points)

            for layer_idx in layers:
                runtime = runtimes[layer_idx]
                sae: SparseDictionary = runtime["sae"]
                acts = cache[runtime["hook_point_in"]]
                with torch.no_grad():
                    feature_acts = sae.encode(acts)
                structured_batch, sentence_feature_rows = self._collect_layer_structured_batch(
                    tokens=tokens,
                    token_mask=token_mask,
                    feature_acts=feature_acts,
                    start=start,
                    retain_sentence_activations=retain_sentence_activations,
                )
                structured_data_by_layer[layer_idx].extend(structured_batch)
                if retain_sentence_activations:
                    sentence_rows_by_layer[layer_idx].extend(sentence_feature_rows)

        layer_outputs: Dict[int, Dict[str, Any]] = {}
        for layer_idx in layers:
            layer_outputs[layer_idx] = {
                "layer_stats": compute_layer_stats(structured_data_by_layer[layer_idx])
            }
            if retain_sentence_activations:
                layer_outputs[layer_idx]["sentence_feature_rows"] = sentence_rows_by_layer[
                    layer_idx
                ]

        return layer_outputs

    def analyze_feature(
        self,
        feature_file: str,
        layers: List[int],
        top_k: int = 10,
    ) -> Dict[str, Any]:
        lines = load_text_data(feature_file)
        results = {
            "feature_file": feature_file,
            "total_examples": len(lines),
            "layers_analyzed": layers,
            "top_k": top_k,
            "layer_results": {},
            "unified_results": [],
        }

        progress = ProgressLogger(len(layers), "Analyzing layers")
        feature_name = os.path.splitext(os.path.basename(feature_file))[0]
        analysis_by_layer = self._analyze_multiple_layers(lines, layers)

        for layer_position, layer_idx in enumerate(layers, start=1):
            print(
                f"[layer {layer_position}/{len(layers)}] finalize: "
                f"layer={layer_idx}, top_k={top_k}"
            )
            layer_start = time.perf_counter()
            try:
                layer_stats = analysis_by_layer[layer_idx]["layer_stats"]
                if layer_stats:
                    top_features = get_top_features_by_frc(layer_stats, top_k)
                    results["layer_results"][layer_idx] = {
                        "total_base_vectors": len(layer_stats),
                        "top_features": top_features,
                        "full_stats": layer_stats,
                    }
                    for base_vector, _ in top_features[:3]:
                        stats = layer_stats[base_vector]
                        results["unified_results"].append(
                            {
                                "feature": feature_name,
                                "layer": layer_idx,
                                "base_vector": int(base_vector),
                                "ps": stats["ps"],
                                "pn": stats["pn"],
                                "frc": stats["frc"],
                                "avg_max_activation": stats["avg_max_activation"],
                            }
                        )
                    top_frc = float(top_features[0][1]) if top_features else float("nan")
                    print(
                        f"[layer {layer_position}/{len(layers)}] done: "
                        f"layer={layer_idx}, base_vectors={len(layer_stats)}, "
                        f"top_frc={top_frc:.4f}, elapsed={time.perf_counter() - layer_start:.1f}s"
                    )
                else:
                    print(
                        f"[layer {layer_position}/{len(layers)}] done: "
                        f"layer={layer_idx}, base_vectors=0, "
                        f"elapsed={time.perf_counter() - layer_start:.1f}s"
                    )
                progress.update()
            except Exception as exc:
                print(f"Error analyzing layer {layer_idx}: {exc}")
                results["layer_results"][layer_idx] = {"error": str(exc)}
                progress.update()

        progress.finish()
        return results

    def _analyze_single_layer(
        self,
        lines: List[str],
        layer_idx: int,
        retain_sentence_activations: bool = False,
    ) -> Dict[str, Any] | Dict[int, Dict[str, float]]:
        layer_output = self._analyze_multiple_layers(
            lines,
            [layer_idx],
            retain_sentence_activations=retain_sentence_activations,
        )[layer_idx]
        if retain_sentence_activations:
            return layer_output
        return layer_output["layer_stats"]

    def clear_cache(self):
        self._model_cache.clear()
        self._base_model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Batch-analyze feature text files with LlamaScopeLinguisticAnalyzer."
        )
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Base model name for TransformerLens HookedTransformer.from_pretrained().",
    )
    parser.add_argument(
        "--sae-path-template",
        required=True,
        help="SparseDictionary template, e.g. org/repo:.../layer{}_transcoder_...",
    )
    parser.add_argument(
        "--feature-files",
        nargs="+",
        required=True,
        help="One or more feature text files (alternating original/minimal pairs).",
    )
    parser.add_argument(
        "--layers",
        required=True,
        help="Comma-separated layer indices, e.g. 0,7,15,19,25,31",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save batch analysis results.",
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

    parser.set_defaults(prepend_bos=True)
    args = parser.parse_args()

    layers = _parse_layers(args.layers)
    analyzer = LlamaScopeLinguisticAnalyzer(
        model_path=args.model_path,
        sae_path_template=args.sae_path_template,
        device=args.device,
        batch_size=args.batch_size,
        torch_dtype=_parse_torch_dtype(args.torch_dtype),
        prepend_bos=args.prepend_bos,
        fold_activation_scale=args.fold_activation_scale,
    )

    try:
        results = analyzer.batch_analyze_features(
            feature_files=args.feature_files,
            layers=layers,
            output_dir=args.output_dir,
            top_k=args.top_k,
        )
        print("Batch analysis complete.")
        print(f"Output directory: {args.output_dir}")
        print(
            f"Successful: {results.get('successful_analyses', 0)}, "
            f"Failed: {results.get('failed_analyses', 0)}"
        )
    finally:
        analyzer.clear_cache()


if __name__ == "__main__":
    main()
