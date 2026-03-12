"""
Model intervention module for train.py SAE checkpoints.

This module mirrors LinguaLens.lingualens.intervener.Intervener but uses
OpenSAE-style intervention utilities implemented in local `util.py`.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from glob import glob
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, PreTrainedModel

from LinguaLens.lingualens.utils import (
    ProgressLogger,
    get_available_device,
    save_json_results,
    setup_tokenizer,
)
from util import InterventionConfig, TransformerWithSae


class Intervener:
    """
    Intervener compatible with train.py SAE checkpoints.

    Internally this class delegates intervention behavior to
    util.TransformerWithSae (ported from OpenSAE style logic).
    """

    _shared_transformers: Dict[Tuple[str, str, str], PreTrainedModel] = {}
    _shared_tokenizers: Dict[str, Any] = {}

    def __init__(
        self,
        model_path: str,
        sae_path: str,
        device: Optional[str] = None,
        layer_idx: Optional[int] = None,
        k: int = 32,
        normalization: str = "Scalar",
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        self.model_path = model_path
        self.sae_path = sae_path
        self.device = device or get_available_device()
        self.layer_idx = layer_idx if layer_idx is not None else self._infer_layer_idx(sae_path)
        self.k = int(k)
        self.normalization = normalization
        self.torch_dtype = torch_dtype
        self.tokenizer = self._get_or_load_tokenizer(model_path)

        self.model: Optional[TransformerWithSae] = None
        self._intervention_config = InterventionConfig(
            intervention=False,
            intervention_indices=[],
            intervention_value=0.0,
        )

        self._load_models()

    @classmethod
    def _get_or_load_tokenizer(cls, model_path: str):
        if model_path not in cls._shared_tokenizers:
            cls._shared_tokenizers[model_path] = setup_tokenizer(model_path)
        return cls._shared_tokenizers[model_path]

    @classmethod
    def _get_transformer_cache_key(
        cls,
        model_path: str,
        device: str,
        torch_dtype: Optional[torch.dtype],
    ) -> Tuple[str, str, str]:
        dtype_name = str(torch_dtype) if torch_dtype is not None else "none"
        return (model_path, device, dtype_name)

    @classmethod
    def _get_or_load_shared_transformer(
        cls,
        model_path: str,
        device: str,
        torch_dtype: Optional[torch.dtype],
    ) -> PreTrainedModel:
        cache_key = cls._get_transformer_cache_key(model_path, device, torch_dtype)
        if cache_key in cls._shared_transformers:
            return cls._shared_transformers[cache_key]

        print(f"Loading transformer model from {model_path}")
        try:
            transformer = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
            )
        except Exception:
            transformer = AutoModelForCausalLM.from_pretrained(model_path)

        transformer = transformer.to(device)
        if torch_dtype is not None:
            try:
                transformer = transformer.to(dtype=torch_dtype)
            except Exception:
                pass
        transformer.eval()
        cls._shared_transformers[cache_key] = transformer
        return transformer

    @staticmethod
    def _infer_layer_idx(sae_path: str) -> int:
        name = os.path.basename(sae_path)
        match = re.search(r"layer(\d+)", name)
        if not match:
            raise ValueError(
                "Could not infer layer index from SAE filename. "
                "Pass layer_idx explicitly, e.g. layer_idx=12."
            )
        return int(match.group(1))

    def _load_models(self):
        if not os.path.isfile(self.sae_path):
            raise FileNotFoundError(f"SAE checkpoint not found: {self.sae_path}")

        print(f"Loading SAE from {self.sae_path}")
        shared_transformer = self._get_or_load_shared_transformer(
            self.model_path,
            self.device,
            self.torch_dtype,
        )

        self.model = TransformerWithSae(
            transformer=shared_transformer,
            sae=self.sae_path,
            device=self.device,
            intervention_config=self._intervention_config,
            layer_idx=self.layer_idx,
            normalization=self.normalization,
            k=self.k,
        )

        self.model.transformer.eval()
        print("Models loaded successfully!")

    def update_intervention_config(self, config: InterventionConfig):
        self._intervention_config = config
        if self.model is not None:
            self.model.update_intervention_config(config)

    def _generate_with_intervention(
        self,
        inputs: Dict[str, torch.Tensor],
        max_new_tokens: int,
        temperature: float,
    ) -> str:
        try:
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True if temperature > 0 else False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            output_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            return output_text
        except Exception as exc:
            return f"[GENERATION ERROR]: {str(exc)}"

    def run_intervention_experiment(
        self,
        input_prompt: str,
        intervention_indices: List[int],
        output_path: str,
        num_generations: int = 10,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        experiment_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        input_prompts = _split_prompts(input_prompt)
        experiment_results = {
            "experiment_name": experiment_name or "intervention_experiment",
            "input_prompt": input_prompt,
            "input_prompts": input_prompts,
            "intervention_indices": intervention_indices,
            "num_generations": num_generations,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "layer_idx": self.layer_idx,
            "prompt_results": [],
        }

        output_dir = os.path.dirname(output_path) or "."
        os.makedirs(output_dir, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as out_file:
            out_file.write(f"Experiment: {experiment_results['experiment_name']}\n\n")
            out_file.write("[INPUT PROMPTS]:\n")
            for prompt_idx, prompt in enumerate(input_prompts, start=1):
                out_file.write(f"{prompt_idx}. {prompt}\n")
            out_file.write("\n")
            out_file.write(f"[INTERVENTION INDICES]: {intervention_indices}\n\n")
            out_file.write(f"[LAYER IDX]: {self.layer_idx}\n\n")

        progress = ProgressLogger(
            len(input_prompts) * num_generations * 3,
            "Running intervention experiment",
        )
        encoded_prompts = [
            self.tokenizer(prompt, return_tensors="pt").to(self.device)
            for prompt in input_prompts
        ]

        with open(output_path, "a", encoding="utf-8") as out_file:
            for prompt_idx, (prompt, inputs) in enumerate(
                zip(input_prompts, encoded_prompts), start=1
            ):
                prompt_result = {
                    "prompt_index": prompt_idx,
                    "input_prompt": prompt,
                    "conditions": {"ablation": [], "enhancement": [], "control": []},
                }
                prompt_lines = [f"=== Prompt {prompt_idx} ===\n", prompt + "\n\n"]

                for generation in range(1, num_generations + 1):
                    print(f"Starting prompt {prompt_idx}, generation {generation}...")
                    prompt_lines.append(
                        f"--- Prompt {prompt_idx} / Generation {generation} ---\n"
                    )
                    ablation_config = InterventionConfig(
                        intervention=True,
                        intervention_mode="set",
                        intervention_indices=intervention_indices,
                        intervention_value=0.0,
                        prompt_only=False,
                    )
                    self.update_intervention_config(ablation_config)
                    ablation_output = self._generate_with_intervention(
                        inputs, max_new_tokens, temperature
                    )
                    prompt_lines.append("[Ablation (set value=0)]:\n")
                    prompt_lines.append(ablation_output + "\n\n")
                    progress.update()

                    enhancement_config = InterventionConfig(
                        intervention=True,
                        intervention_mode="set",
                        intervention_indices=intervention_indices,
                        intervention_value=10.0,
                        prompt_only=False,
                    )
                    self.update_intervention_config(enhancement_config)
                    enhancement_output = self._generate_with_intervention(
                        inputs, max_new_tokens, temperature
                    )
                    prompt_lines.append("[Enhancement (set value=10)]:\n")
                    prompt_lines.append(enhancement_output + "\n\n")
                    progress.update()

                    control_config = InterventionConfig(
                        intervention=True,
                        intervention_mode="multiply",
                        intervention_indices=intervention_indices,
                        intervention_value=1.0,
                        prompt_only=False,
                    )
                    self.update_intervention_config(control_config)
                    control_output = self._generate_with_intervention(
                        inputs, max_new_tokens, temperature
                    )
                    prompt_lines.append("[Control (multiply value=1)]:\n")
                    prompt_lines.append(control_output + "\n\n")
                    progress.update()

                    prompt_result["conditions"]["ablation"].append(ablation_output)
                    prompt_result["conditions"]["enhancement"].append(enhancement_output)
                    prompt_result["conditions"]["control"].append(control_output)
                    print(
                        f"Prompt {prompt_idx}, generation {generation} completed and buffered "
                        f"for {output_path}."
                    )

                out_file.write("".join(prompt_lines))
                out_file.flush()
                experiment_results["prompt_results"].append(prompt_result)

        if len(experiment_results["prompt_results"]) == 1:
            experiment_results["conditions"] = experiment_results["prompt_results"][0][
                "conditions"
            ]

        progress.finish()

        results_json_path = output_path.replace(".txt", "_results.json")
        save_json_results(experiment_results, results_json_path)

        print(f"\nExperiment complete! Results saved to {output_path}")
        print(f"Structured results saved to {results_json_path}")
        return experiment_results

    def batch_intervention_experiments(
        self,
        input_files: List[str],
        intervention_configs: List[Dict[str, Any]],
        output_dir: str,
        num_generations: int = 5,
    ) -> Dict[str, Any]:
        os.makedirs(output_dir, exist_ok=True)

        batch_results = {
            "total_experiments": len(input_files) * len(intervention_configs),
            "successful_experiments": 0,
            "failed_experiments": 0,
            "experiment_summaries": {},
        }

        progress = ProgressLogger(
            len(input_files) * len(intervention_configs),
            "Batch intervention experiments",
        )

        for input_file in input_files:
            with open(input_file, "r", encoding="utf-8") as f:
                input_prompt = f.read().strip()
            input_name = os.path.splitext(os.path.basename(input_file))[0]

            for config in intervention_configs:
                experiment_name = f"{input_name}_{config['name']}"
                try:
                    output_path = os.path.join(output_dir, f"{experiment_name}.txt")
                    self.run_intervention_experiment(
                        input_prompt=input_prompt,
                        intervention_indices=config["intervention_indices"],
                        output_path=output_path,
                        num_generations=num_generations,
                        max_new_tokens=config.get("max_new_tokens", 100),
                        temperature=config.get("temperature", 1.0),
                        experiment_name=experiment_name,
                    )
                    batch_results["experiment_summaries"][experiment_name] = {
                        "input_file": input_file,
                        "config": config,
                        "output_path": output_path,
                        "status": "success",
                    }
                    batch_results["successful_experiments"] += 1
                except Exception as exc:
                    print(f"Failed experiment {experiment_name}: {exc}")
                    batch_results["experiment_summaries"][experiment_name] = {
                        "input_file": input_file,
                        "config": config,
                        "status": "failed",
                        "error": str(exc),
                    }
                    batch_results["failed_experiments"] += 1
                finally:
                    progress.update()

        progress.finish()
        summary_path = os.path.join(output_dir, "batch_intervention_summary.json")
        save_json_results(batch_results, summary_path)
        print(f"\nBatch intervention complete! Summary saved to {summary_path}")
        return batch_results

    def quick_intervention_test(
        self,
        input_prompt: str,
        intervention_indices: List[int],
        intervention_values: List[float] = [0.0, 1.0, 5.0, 10.0],
    ) -> Dict[str, str]:
        results = {}
        inputs = self.tokenizer(input_prompt, return_tensors="pt").to(self.device)

        for value in intervention_values:
            config = InterventionConfig(
                intervention=True,
                intervention_mode="set",
                intervention_indices=intervention_indices,
                intervention_value=float(value),
                prompt_only=False,
            )
            self.update_intervention_config(config)
            output = self._generate_with_intervention(
                inputs, max_new_tokens=50, temperature=0.7
            )
            results[f"value_{value}"] = output
            print(f"Intervention value {value}: {output[:100]}...")

        return results

    def clear_cache(self):
        self._intervention_config = InterventionConfig(
            intervention=False,
            intervention_indices=[],
            intervention_value=0.0,
        )
        if self.model is not None:
            self.model.update_intervention_config(self._intervention_config)
            self.model.remove_hooks()
        self.model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _parse_torch_dtype(dtype_name: str) -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")
    return mapping[dtype_name]


def _split_prompts(prompt_text: str) -> List[str]:
    prompts = [line.strip() for line in prompt_text.splitlines() if line.strip()]
    if not prompts:
        raise ValueError("Prompt text is empty.")
    return prompts


def _collect_crosslayer_json_paths(crosslayer_path: str) -> List[str]:
    if os.path.isdir(crosslayer_path):
        paths = sorted(glob(os.path.join(crosslayer_path, "*_layer*_evolution.json")))
        if not paths:
            raise FileNotFoundError(
                f"No per-layer crosslayer JSON files found in directory: {crosslayer_path}"
            )
        return paths

    if not os.path.exists(crosslayer_path):
        raise FileNotFoundError(f"Crosslayer input not found: {crosslayer_path}")
    return [crosslayer_path]


def _load_crosslayer_jsons(crosslayer_path: str) -> List[Tuple[str, Dict[str, Any]]]:
    loaded = []
    for path in _collect_crosslayer_json_paths(crosslayer_path):
        with open(path, "r", encoding="utf-8") as f:
            loaded.append((path, json.load(f)))
    return loaded


def _select_best_intervention_candidate(crosslayer_results: Dict[str, Any]) -> Dict[str, Any]:
    if "layer_candidates" in crosslayer_results:
        best = None
        for layer_str, info in crosslayer_results.get("layer_candidates", {}).items():
            if info.get("status") != "ready":
                continue
            candidate = {
                "base_vectors": [
                    int(base_vec) for base_vec in info.get("base_vectors", [info["base_vector"]])
                ],
                "layer_idx": int(info.get("layer_idx", layer_str)),
                "frc": float(info["frc"]),
            }
            if best is None or candidate["frc"] > best["frc"]:
                best = candidate
        if best is not None:
            return best

    best = None
    base_vector_evolution = (
        crosslayer_results.get("evolution_data", {}).get("base_vector_evolution", {})
    )

    for base_vec_str, layer_map in base_vector_evolution.items():
        for layer_str, stats in layer_map.items():
            try:
                frc = float(stats.get("frc", float("-inf")))
                base_vec = int(base_vec_str)
                layer_idx = int(layer_str)
            except Exception:
                continue

            if best is None or frc > best["frc"]:
                best = {
                    "base_vectors": [base_vec],
                    "layer_idx": layer_idx,
                    "frc": frc,
                }

    if best is not None:
        layer_results = crosslayer_results.get("base_results", {}).get("layer_results", {})
        result = layer_results.get(best["layer_idx"], layer_results.get(str(best["layer_idx"]), {}))
        top_features = result.get("top_features", [])
        if top_features:
            best["base_vectors"] = [int(base_vec) for base_vec, _ in top_features[:3]]
        return best

    layer_results = crosslayer_results.get("base_results", {}).get("layer_results", {})
    for layer_str, result in layer_results.items():
        top_features = result.get("top_features", [])
        if not top_features:
            continue
        base_vec, frc = top_features[0]
        return {
            "base_vectors": [int(curr_base_vec) for curr_base_vec, _ in top_features[:3]],
            "layer_idx": int(layer_str),
            "frc": float(frc),
        }

    raise RuntimeError("No intervention candidate found in crosslayer results.")


def _select_best_intervention_candidate_from_many(
    crosslayer_entries: List[Tuple[str, Dict[str, Any]]]
) -> Dict[str, Any]:
    best = None
    for path, crosslayer_results in crosslayer_entries:
        candidate = _select_best_intervention_candidate(crosslayer_results)
        candidate["cross_json"] = path
        if best is None or candidate["frc"] > best["frc"]:
            best = candidate
    if best is None:
        raise RuntimeError("No intervention candidate found in crosslayer inputs.")
    return best


def _select_per_layer_intervention_candidates(
    crosslayer_results: Dict[str, Any]
) -> List[Dict[str, Any]]:
    if "layer_candidates" in crosslayer_results:
        candidates = []
        for layer_str, info in sorted(
            crosslayer_results.get("layer_candidates", {}).items(),
            key=lambda item: int(item[0]),
        ):
            if info.get("status") != "ready":
                continue
            candidates.append(
                {
                    "base_vectors": [
                        int(base_vec)
                        for base_vec in info.get("base_vectors", [info["base_vector"]])
                    ],
                    "layer_idx": int(info.get("layer_idx", layer_str)),
                    "frc": float(info["frc"]),
                }
            )
        if not candidates:
            raise RuntimeError("No per-layer intervention candidates found in layer-candidate summary.")
        return candidates

    layer_results = crosslayer_results.get("base_results", {}).get("layer_results", {})
    candidates: List[Dict[str, Any]] = []

    sortable_layers = []
    for key in layer_results.keys():
        try:
            sortable_layers.append(int(key))
        except Exception:
            continue

    for layer_idx in sorted(set(sortable_layers)):
        result = layer_results.get(layer_idx, layer_results.get(str(layer_idx), {}))
        top_features = result.get("top_features", [])
        if not top_features:
            continue
        base_vec, frc = top_features[0]
        candidates.append(
            {
                "base_vectors": [int(curr_base_vec) for curr_base_vec, _ in top_features[:3]],
                "layer_idx": int(layer_idx),
                "frc": float(frc),
            }
        )

    if not candidates:
        raise RuntimeError("No per-layer intervention candidates found in crosslayer results.")

    return candidates


def _select_per_layer_intervention_candidates_from_many(
    crosslayer_entries: List[Tuple[str, Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    candidates = []
    for path, crosslayer_results in crosslayer_entries:
        per_file_candidates = _select_per_layer_intervention_candidates(crosslayer_results)
        if len(per_file_candidates) != 1:
            raise RuntimeError(
                "Per-layer crosslayer file should contain exactly one layer result: "
                f"{path}"
            )
        candidate = per_file_candidates[0]
        candidate["cross_json"] = path
        candidates.append(candidate)

    if not candidates:
        raise RuntimeError("No per-layer intervention candidates found in crosslayer inputs.")

    candidates.sort(key=lambda item: int(item["layer_idx"]))
    return candidates


def _save_selection_metadata(
    output_path: str,
    selection_meta: Dict[str, Any],
) -> str:
    selection_path = output_path.replace(".txt", "_selection.json")
    output_dir = os.path.dirname(output_path) or "."
    os.makedirs(output_dir, exist_ok=True)
    with open(selection_path, "w", encoding="utf-8") as f:
        json.dump(selection_meta, f, ensure_ascii=False, indent=2)
    return selection_path


def _get_result_paths(output_path: str) -> Dict[str, str]:
    return {
        "output_path": output_path,
        "results_json": output_path.replace(".txt", "_results.json"),
        "selection_json": output_path.replace(".txt", "_selection.json"),
    }


def _is_completed_run(output_path: str) -> bool:
    paths = _get_result_paths(output_path)
    return all(os.path.exists(path) for path in paths.values())


def _load_existing_selection(output_path: str) -> Optional[Dict[str, Any]]:
    selection_path = _get_result_paths(output_path)["selection_json"]
    if not os.path.exists(selection_path):
        return None
    with open(selection_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _run_single_candidate(
    args: argparse.Namespace,
    candidate: Dict[str, Any],
    prompt_text: str,
    output_path: str,
    experiment_name: Optional[str] = None,
) -> Dict[str, Any]:
    if args.resume and _is_completed_run(output_path):
        print(f"Skipping completed run: {output_path}")
        existing = _load_existing_selection(output_path)
        if existing is not None:
            existing["resume_status"] = "skipped_completed"
            return existing
        return {
            "cross_json": candidate.get("cross_json", args.crosslayer_json),
            "selected_layer_idx": candidate["layer_idx"],
            "selected_base_vectors": candidate["base_vectors"],
            "selected_frc": candidate["frc"],
            "sae_path": args.sae_path_template.format(candidate["layer_idx"]),
            "resume_status": "skipped_completed",
        }

    if args.resume:
        paths = _get_result_paths(output_path)
        existing_paths = [path for path in paths.values() if os.path.exists(path)]
        if existing_paths:
            print(
                f"Resuming incomplete run by overwriting partial outputs: {output_path}"
            )

    sae_path = args.sae_path_template.format(candidate["layer_idx"])
    input_prompts = _split_prompts(prompt_text)

    selection_meta = {
        "cross_json": candidate.get("cross_json", args.crosslayer_json),
        "selected_layer_idx": candidate["layer_idx"],
        "selected_base_vectors": candidate["base_vectors"],
        "selected_frc": candidate["frc"],
        "sae_path": sae_path,
        "input_prompts": input_prompts,
        "resume_status": "executed",
    }
    selection_path = _save_selection_metadata(output_path, selection_meta)

    intervener = Intervener(
        model_path=args.model_path,
        sae_path=sae_path,
        device=args.device,
        layer_idx=candidate["layer_idx"],
        k=args.k,
        normalization=args.normalization,
        torch_dtype=_parse_torch_dtype(args.torch_dtype),
    )

    try:
        resolved_experiment_name = experiment_name or (
            f"intervention_layer{candidate['layer_idx']}_bv{'-'.join(map(str, candidate['base_vectors']))}"
        )
        intervener.run_intervention_experiment(
            input_prompt=prompt_text,
            intervention_indices=candidate["base_vectors"],
            output_path=output_path,
            num_generations=args.num_generations,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            experiment_name=resolved_experiment_name,
        )
    finally:
        intervener.clear_cache()

    print(
        "Selected intervention target: "
        f"layer={candidate['layer_idx']}, base_vectors={candidate['base_vectors']}, frc={candidate['frc']}"
    )
    print(f"Selection metadata saved to {selection_path}")
    return selection_meta


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run an intervention experiment using the best feature selected "
            "from crosslayer_lingualens.py output."
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
        "--crosslayer-json",
        required=True,
        help="Path to a per-layer crosslayer JSON file or a directory containing *_layer*_evolution.json files.",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Path to save the intervention result text file.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save intervention result files in per-layer mode.",
    )
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument(
        "--prompt",
        help="Test text used as the intervention prompt. Multiple lines are supported.",
    )
    prompt_group.add_argument(
        "--prompt-file",
        help="Path to a text file. Each non-empty line is treated as one prompt.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device override (e.g. cuda:0 or cpu). Default: auto.",
    )
    parser.add_argument("--k", type=int, default=32, help="SAE top-k activation setting.")
    parser.add_argument("--normalization", default="Scalar", help="Activation normalization mode.")
    parser.add_argument(
        "--torch-dtype",
        choices=["bfloat16", "float16", "float32"],
        default="bfloat16",
        help="Torch dtype for loading the base model.",
    )
    parser.add_argument(
        "--num-generations",
        type=int,
        default=5,
        help="Number of generations per intervention condition.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=100,
        help="Maximum number of newly generated tokens.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--experiment-name",
        default=None,
        help="Optional experiment name. Default: derived from selected layer/base vector.",
    )
    parser.add_argument(
        "--selection-mode",
        choices=["best", "per-layer"],
        default="best",
        help="How to select intervention targets from crosslayer_json.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip runs whose output txt/results/selection files already exist.",
    )

    args = parser.parse_args()
    if args.selection_mode == "best" and not args.output_path:
        raise ValueError("--output-path is required when --selection-mode=best.")
    if args.selection_mode == "per-layer" and not args.output_dir:
        raise ValueError("--output-dir is required when --selection-mode=per-layer.")

    crosslayer_entries = _load_crosslayer_jsons(args.crosslayer_json)

    prompt_text = args.prompt
    if args.prompt_file:
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            prompt_text = f.read()

    if args.selection_mode == "best":
        best = _select_best_intervention_candidate_from_many(crosslayer_entries)
        _run_single_candidate(
            args=args,
            candidate=best,
            prompt_text=prompt_text,
            output_path=args.output_path,
            experiment_name=args.experiment_name,
        )
        return 0

    os.makedirs(args.output_dir, exist_ok=True)
    candidates = _select_per_layer_intervention_candidates_from_many(crosslayer_entries)

    for candidate in candidates:
        output_path = os.path.join(
            args.output_dir, f"layer{candidate['layer_idx']}_intervention.txt"
        )
        experiment_name = args.experiment_name or (
            f"intervention_layer{candidate['layer_idx']}_bv{'-'.join(map(str, candidate['base_vectors']))}"
        )
        _run_single_candidate(
            args=args,
            candidate=candidate,
            prompt_text=prompt_text,
            output_path=output_path,
            experiment_name=experiment_name,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
