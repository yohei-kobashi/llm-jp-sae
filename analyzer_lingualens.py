"""
LinguisticAnalyzer extension for SAEs trained by this repository's train.py.

This module provides a subclass of LinguaLens's LinguisticAnalyzer that can load
checkpoint files saved as:
    sae_layer{layer}.pth
and run the same PS/PN/FRC analysis pipeline.
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM

from config import SaeConfig
from model import SimpleHook, SparseAutoEncoder, normalize_activation
from LinguaLens.lingualens.metrics import compute_layer_stats, get_top_features_by_frc
from LinguaLens.lingualens.utils import (
    ProgressLogger,
    load_text_data,
    save_json_results,
    setup_tokenizer,
    get_available_device,
    validate_layer_indices,
)


class BaseTrainSaeLinguisticAnalyzer:
    """
    Minimal analyzer base that avoids OpenSAE imports.

    This mirrors only the LinguaLens functionality needed by the local
    train.py-compatible analyzer.
    """

    def __init__(
        self,
        model_path: str,
        sae_path_template: str,
        device: Optional[str] = None,
    ):
        self.model_path = model_path
        self.sae_path_template = sae_path_template
        self.device = device or get_available_device()
        self.tokenizer = setup_tokenizer(model_path)
        self._model_cache: Dict[int, Any] = {}

    def _build_layer_analysis_result(
        self,
        feature_file: str,
        total_examples: int,
        layer_idx: int,
        top_k: int,
        layer_stats: Dict[int, Dict[str, float]],
        sentence_feature_rows: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        feature_name = os.path.splitext(os.path.basename(feature_file))[0]
        layer_result: Dict[str, Any] = {
            "feature_file": feature_file,
            "total_examples": total_examples,
            "layers_analyzed": [layer_idx],
            "top_k": top_k,
            "layer_results": {},
            "unified_results": [],
        }
        if sentence_feature_rows is not None:
            layer_result["sentence_feature_rows"] = sentence_feature_rows

        if layer_stats:
            top_features = get_top_features_by_frc(layer_stats, top_k)
            layer_result["layer_results"][layer_idx] = {
                "total_base_vectors": len(layer_stats),
                "top_features": top_features,
                "full_stats": layer_stats,
            }
            for base_vector, _ in top_features[:3]:
                stats = layer_stats[base_vector]
                layer_result["unified_results"].append(
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

        return layer_result

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

        for layer_position, layer_idx in enumerate(layers, start=1):
            print(
                f"[layer {layer_position}/{len(layers)}] start: "
                f"layer={layer_idx}, top_k={top_k}"
            )
            layer_start = time.perf_counter()
            try:
                layer_stats = self._analyze_single_layer(lines, layer_idx)
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

    def analyze_feature_layer(
        self,
        feature_file: str,
        layer_idx: int,
        top_k: int = 10,
        layer_position: Optional[int] = None,
        total_layers: Optional[int] = None,
        retain_sentence_activations: bool = False,
    ) -> Dict[str, Any]:
        lines = load_text_data(feature_file)
        layer_label = (
            f"[layer {layer_position}/{total_layers}]"
            if layer_position is not None and total_layers is not None
            else "[layer]"
        )
        print(f"{layer_label} start: layer={layer_idx}, top_k={top_k}")
        layer_start = time.perf_counter()

        analysis_output = self._analyze_single_layer(
            lines,
            layer_idx,
            retain_sentence_activations=retain_sentence_activations,
        )
        if retain_sentence_activations:
            layer_stats = analysis_output["layer_stats"]
            sentence_feature_rows = analysis_output["sentence_feature_rows"]
        else:
            layer_stats = analysis_output
            sentence_feature_rows = []
        layer_result = self._build_layer_analysis_result(
            feature_file=feature_file,
            total_examples=len(lines),
            layer_idx=layer_idx,
            top_k=top_k,
            layer_stats=layer_stats,
            sentence_feature_rows=sentence_feature_rows if retain_sentence_activations else None,
        )
        if layer_stats:
            top_features = layer_result["layer_results"][layer_idx]["top_features"]
            top_frc = float(top_features[0][1]) if top_features else float("nan")
            print(
                f"{layer_label} done: layer={layer_idx}, base_vectors={len(layer_stats)}, "
                f"top_frc={top_frc:.4f}, elapsed={time.perf_counter() - layer_start:.1f}s"
            )
        else:
            print(
                f"{layer_label} done: layer={layer_idx}, base_vectors=0, "
                f"elapsed={time.perf_counter() - layer_start:.1f}s"
            )

        return layer_result

    def batch_analyze_features(
        self,
        feature_files: List[str],
        layers: List[int],
        output_dir: str,
        top_k: int = 10,
    ) -> Dict[str, Any]:
        os.makedirs(output_dir, exist_ok=True)

        batch_results = {
            "total_features": len(feature_files),
            "layers_analyzed": layers,
            "successful_analyses": 0,
            "failed_analyses": 0,
            "unified_results": [],
            "feature_summaries": {},
        }

        progress = ProgressLogger(len(feature_files), "Batch analysis")

        for feature_file in feature_files:
            try:
                feature_results = self.analyze_feature(feature_file, layers, top_k)
                feature_name = os.path.splitext(os.path.basename(feature_file))[0]
                output_file = os.path.join(output_dir, f"{feature_name}_analysis.json")
                save_json_results(feature_results, output_file)

                batch_results["unified_results"].extend(feature_results["unified_results"])
                batch_results["feature_summaries"][feature_name] = {
                    "total_examples": feature_results["total_examples"],
                    "layers_completed": len(
                        [
                            l
                            for l in layers
                            if l in feature_results["layer_results"]
                            and "error" not in feature_results["layer_results"][l]
                        ]
                    ),
                }
                batch_results["successful_analyses"] += 1
            except Exception as exc:
                print(f"Failed to analyze {feature_file}: {exc}")
                batch_results["failed_analyses"] += 1
            finally:
                progress.update()

        progress.finish()
        summary_file = os.path.join(output_dir, "batch_summary.json")
        save_json_results(batch_results, summary_file)
        return batch_results


class TrainSaeLinguisticAnalyzer(BaseTrainSaeLinguisticAnalyzer):
    """
    LinguisticAnalyzer compatible with train.py checkpoints.

    Unlike the base class (which expects OpenSAE format), this class loads
    `sae_layer{layer}.pth` produced by train.py and constructs structured data
    directly from model hooks and SAE outputs.
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
        super().__init__(
            model_path=model_path,
            sae_path_template=sae_path_template,
            device=device,
        )
        self.k = int(k)
        self.normalization = normalization
        self.batch_size = int(batch_size)
        self.torch_dtype = torch_dtype
        self._base_model = None
        self._layer_modules: Optional[Dict[str, Any]] = None

    def _load_base_model(self):
        if self._base_model is not None:
            return self._base_model

        try:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=self.torch_dtype,
            )
        except Exception:
            model = AutoModelForCausalLM.from_pretrained(self.model_path)

        model = model.to(self.device)
        model.eval()
        self._base_model = model
        return model

    def _tokenize_batch(
        self,
        batch_lines: List[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        enc = self.tokenizer(
            batch_lines,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        return input_ids, attention_mask.to(self.device)

    def _extract_hook_activations(
        self,
        hook: SimpleHook,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        acts = hook.output
        if isinstance(acts, tuple):
            acts = acts[0]
        if acts.dim() == 2:
            acts = acts.unsqueeze(0)
        return acts[:, 1:, :], attention_mask[:, 1:]

    def _resolve_layer_modules(self) -> Dict[str, Any]:
        if self._layer_modules is not None:
            return self._layer_modules

        model = self._load_base_model()
        core = getattr(model, "model", None) or getattr(model, "transformer", None)
        if core is None:
            raise ValueError(
                "Unsupported model structure: expected `model` or `transformer` module."
            )

        embed_tokens = getattr(core, "embed_tokens", None) or getattr(core, "wte", None)
        layers = getattr(core, "layers", None) or getattr(core, "h", None)
        if embed_tokens is None or layers is None:
            raise ValueError(
                "Unsupported model structure: expected `embed_tokens/wte` and `layers/h`."
            )

        self._layer_modules = {"embed_tokens": embed_tokens, "layers": layers}
        return self._layer_modules

    def _load_train_sae(self, layer_idx: int) -> SparseAutoEncoder:
        ckpt_path = self.sae_path_template.format(layer_idx)
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"SAE checkpoint not found: {ckpt_path}")

        state = torch.load(ckpt_path, map_location=self.device)

        if "b_dec" in state:
            d_in = int(state["b_dec"].shape[0])
        elif "encoder.weight" in state:
            d_in = int(state["encoder.weight"].shape[1])
        else:
            raise KeyError(
                f"Cannot infer SAE input width from checkpoint: {ckpt_path}"
            )

        if "encoder.weight" in state:
            num_latents = int(state["encoder.weight"].shape[0])
        elif "W_dec" in state:
            num_latents = int(state["W_dec"].shape[0])
        else:
            raise KeyError(
                f"Cannot infer SAE latent width from checkpoint: {ckpt_path}"
            )

        if num_latents % d_in != 0:
            raise ValueError(
                f"Invalid SAE shape in {ckpt_path}: num_latents={num_latents}, d_in={d_in}"
            )

        expansion_factor = num_latents // d_in
        cfg = SaeConfig(d_in=d_in, expansion_factor=expansion_factor, k=self.k)
        sae = SparseAutoEncoder(cfg).to(self.device)
        sae.load_state_dict(state)
        sae.eval()
        return sae

    def _get_sae_model(self, layer_idx: int) -> Dict[str, Any]:
        if layer_idx in self._model_cache:
            return self._model_cache[layer_idx]

        layer_modules = self._resolve_layer_modules()
        target_module = (
            layer_modules["embed_tokens"]
            if layer_idx == 0
            else layer_modules["layers"][layer_idx - 1]
        )

        hook = SimpleHook(target_module)
        sae = self._load_train_sae(layer_idx)
        runtime = {"sae": sae, "hook": hook}
        self._model_cache[layer_idx] = runtime
        return runtime

    def _collect_layer_structured_batch(
        self,
        input_ids: torch.Tensor,
        acts: torch.Tensor,
        token_mask: torch.Tensor,
        sae: SparseAutoEncoder,
        start: int,
        retain_sentence_activations: bool,
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        structured_batch: List[Dict[str, Any]] = []
        sentence_feature_rows: List[Dict[str, Any]] = []

        for i in range(input_ids.size(0)):
            valid_len = int(token_mask[i].sum().item())
            token_ids = input_ids[i, 1 : valid_len + 1]
            tokens = self.tokenizer.convert_ids_to_tokens(token_ids.tolist())
            tokens = [tok.replace("▁", " ") for tok in tokens]

            sentence_tokens: List[Dict[str, Any]] = []
            sentence_latent_activations: Dict[int, float] = {}
            if valid_len > 0:
                activation = acts[i, :valid_len, :]
                activation = normalize_activation(activation, self.normalization)
                activation = activation.to(self.device)

                with torch.no_grad():
                    out = sae(activation)

                top_indices = out.latent_indices.detach().cpu().tolist()
                top_acts = out.latent_acts.detach().cpu().tolist()

                for token, idxs, vals in zip(tokens, top_indices, top_acts):
                    token_activations = []
                    for base_vector, value in zip(idxs, vals):
                        if value > 0:
                            current_value = sentence_latent_activations.get(
                                int(base_vector), 0.0
                            )
                            if value > current_value:
                                sentence_latent_activations[int(base_vector)] = float(value)
                            token_activations.append(
                                {
                                    "base_vector": int(base_vector),
                                    "activation": float(value),
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
            input_ids, attention_mask = self._tokenize_batch(batch_lines)

            with torch.no_grad():
                _ = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )

            for layer_idx in layers:
                runtime = runtimes[layer_idx]
                hook: SimpleHook = runtime["hook"]
                sae: SparseAutoEncoder = runtime["sae"]
                acts, token_mask = self._extract_hook_activations(hook, attention_mask)
                structured_batch, sentence_feature_rows = self._collect_layer_structured_batch(
                    input_ids=input_ids,
                    acts=acts,
                    token_mask=token_mask,
                    sae=sae,
                    start=start,
                    retain_sentence_activations=retain_sentence_activations,
                )
                structured_data_by_layer[layer_idx].extend(structured_batch)
                if retain_sentence_activations:
                    sentence_rows_by_layer[layer_idx].extend(sentence_feature_rows)
                hook.output = None

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
        for runtime in self._model_cache.values():
            hook = runtime.get("hook")
            if hook is not None and hasattr(hook, "hook"):
                try:
                    hook.hook.remove()
                except Exception:
                    pass
        self._model_cache.clear()

        self._base_model = None
        self._layer_modules = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Batch-analyze feature text files with TrainSaeLinguisticAnalyzer "
            "(inherits batch_analyze_features from LinguaLens LinguisticAnalyzer)."
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
    parser.add_argument("--k", type=int, default=32, help="SAE top-k activation setting.")
    parser.add_argument(
        "--normalization",
        default="Scalar",
        help="Activation normalization mode for SAE input.",
    )
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

    layers = _parse_layers(args.layers)
    analyzer = TrainSaeLinguisticAnalyzer(
        model_path=args.model_path,
        sae_path_template=args.sae_path_template,
        device=args.device,
        k=args.k,
        normalization=args.normalization,
        batch_size=args.batch_size,
        torch_dtype=_parse_torch_dtype(args.torch_dtype),
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
