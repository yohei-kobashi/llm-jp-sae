"""
Model intervention module for train.py SAE checkpoints.

This module mirrors LinguaLens.lingualens.intervener.Intervener but uses
OpenSAE-style intervention utilities implemented in local `util.py`.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from glob import glob
from typing import Any, Dict, List, Optional, Tuple

import torch
from openai import OpenAI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, PreTrainedModel

from LinguaLens.lingualens.utils import (
    ProgressLogger,
    get_available_device,
    save_json_results,
    setup_tokenizer,
)
from util import InterventionConfig, TransformerWithSae

JUDGE_MODEL = "gpt-5-mini"


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

    def get_num_latents(self) -> int:
        if self.model is None:
            raise RuntimeError("Model is not loaded.")
        return int(self.model.sae.W_dec.shape[0])

    def _generate_with_intervention(
        self,
        inputs: Dict[str, torch.Tensor],
        max_new_tokens: int,
        temperature: float,
    ) -> str:
        outputs = self._generate_batch_with_intervention(
            inputs=inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        return outputs[0]

    def _generate_batch_with_intervention(
        self,
        inputs: Dict[str, torch.Tensor],
        max_new_tokens: int,
        temperature: float,
    ) -> List[str]:
        try:
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True if temperature > 0 else False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            return [
                self.tokenizer.decode(sequence, skip_special_tokens=True)
                for sequence in generated_ids
            ]
        except Exception as exc:
            batch_size = int(inputs["input_ids"].shape[0])
            return [f"[GENERATION ERROR]: {str(exc)}"] * batch_size

    def run_intervention_experiment(
        self,
        input_prompt: str,
        intervention_indices: List[int],
        output_path: str,
        num_generations: int = 10,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        experiment_name: Optional[str] = None,
        random_seed: int = 0,
        batch_size: int = 8,
        alpha: Optional[float] = None,
        intervention_feature_weights: Optional[List[float]] = None,
    ) -> Dict[str, Any]:
        input_prompts = _split_prompts(input_prompt)
        rng = random.Random(random_seed)
        num_latents = self.get_num_latents()
        batch_size = max(1, int(batch_size))
        experiment_results = {
            "experiment_name": experiment_name or "intervention_experiment",
            "input_prompt": input_prompt,
            "input_prompts": input_prompts,
            "target_intervention_indices": intervention_indices,
            "num_generations": num_generations,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "layer_idx": self.layer_idx,
            "random_seed": random_seed,
            "batch_size": batch_size,
            "alpha": alpha,
            "intervention_feature_weights": intervention_feature_weights,
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
            out_file.write(f"[TARGET INTERVENTION INDICES]: {intervention_indices}\n\n")
            if intervention_feature_weights is not None:
                out_file.write(
                    f"[TARGET INTERVENTION WEIGHTS]: {intervention_feature_weights}\n\n"
                )
            if alpha is not None:
                out_file.write(f"[ALPHA]: {alpha}\n\n")
            out_file.write(f"[LAYER IDX]: {self.layer_idx}\n\n")
            out_file.write(f"[BATCH SIZE]: {batch_size}\n\n")
            out_file.write(
                "[PROMPT RULE]: odd lines -> ablation, even lines -> enhancement\n\n"
            )

        progress = ProgressLogger(
            len(input_prompts) * num_generations * 2,
            "Running intervention experiment",
        )
        prompt_entries = []
        prompt_lines_by_idx: Dict[int, List[str]] = {}
        for prompt_idx, prompt in enumerate(input_prompts, start=1):
            intervention_mode = "ablation" if prompt_idx % 2 == 1 else "enhancement"
            prompt_entries.append(
                {
                    "prompt_index": prompt_idx,
                    "input_prompt": prompt,
                    "intervention_type": intervention_mode,
                    "target_indices": list(intervention_indices),
                    "generations": [],
                }
            )
            prompt_lines_by_idx[prompt_idx] = [
                f"=== Prompt {prompt_idx} ===\n",
                prompt + "\n\n",
                f"[INTERVENTION TYPE]: {intervention_mode}\n",
                f"[TARGET INDICES]: {intervention_indices}\n\n",
            ]

        grouped_entries = {
            "ablation": [entry for entry in prompt_entries if entry["intervention_type"] == "ablation"],
            "enhancement": [
                entry for entry in prompt_entries if entry["intervention_type"] == "enhancement"
            ],
        }
        target_values = {"ablation": 0.0, "enhancement": 10.0}

        for generation in range(1, num_generations + 1):
            for intervention_mode in ("ablation", "enhancement"):
                mode_entries = grouped_entries[intervention_mode]
                if not mode_entries:
                    continue

                target_value = target_values[intervention_mode]
                weighted_intervention_values = None
                if intervention_feature_weights is not None:
                    direction = -1.0 if intervention_mode == "ablation" else 1.0
                    resolved_alpha = 1.0 if alpha is None else float(alpha)
                    weighted_intervention_values = [
                        direction * resolved_alpha * float(weight)
                        for weight in intervention_feature_weights
                    ]
                for batch_start in range(0, len(mode_entries), batch_size):
                    batch_entries = mode_entries[batch_start : batch_start + batch_size]
                    batch_prompts = [entry["input_prompt"] for entry in batch_entries]
                    batch_indices = [entry["prompt_index"] for entry in batch_entries]
                    print(
                        f"Starting {intervention_mode} generation {generation} for prompts "
                        f"{batch_indices}..."
                    )
                    batch_inputs = self.tokenizer(
                        batch_prompts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                    ).to(self.device)

                    target_config = InterventionConfig(
                        intervention=True,
                        intervention_mode="add"
                        if weighted_intervention_values is not None
                        else "set",
                        intervention_indices=intervention_indices,
                        intervention_value=target_value,
                        intervention_weights=weighted_intervention_values,
                        prompt_only=False,
                    )
                    self.update_intervention_config(target_config)
                    target_outputs = self._generate_batch_with_intervention(
                        batch_inputs, max_new_tokens, temperature
                    )
                    progress.update(len(batch_entries))

                    random_indices = _sample_random_intervention_indices(
                        num_latents=num_latents,
                        num_indices=len(intervention_indices),
                        excluded_indices=intervention_indices,
                        rng=rng,
                    )
                    random_config = InterventionConfig(
                        intervention=True,
                        intervention_mode="add"
                        if weighted_intervention_values is not None
                        else "set",
                        intervention_indices=random_indices,
                        intervention_value=target_value,
                        intervention_weights=weighted_intervention_values,
                        prompt_only=False,
                    )
                    self.update_intervention_config(random_config)
                    random_outputs = self._generate_batch_with_intervention(
                        batch_inputs, max_new_tokens, temperature
                    )
                    progress.update(len(batch_entries))

                    for entry, target_output, random_output in zip(
                        batch_entries, target_outputs, random_outputs
                    ):
                        prompt_lines = prompt_lines_by_idx[entry["prompt_index"]]
                        prompt_lines.append(
                            f"--- Prompt {entry['prompt_index']} / Generation {generation} ---\n"
                        )
                        prompt_lines.append(f"[Random INDICES]: {random_indices}\n")
                        if weighted_intervention_values is not None:
                            prompt_lines.append(
                                f"[WEIGHTED INTERVENTION VALUES]: {weighted_intervention_values}\n"
                            )
                        prompt_lines.append(f"[Target {intervention_mode}]:\n")
                        prompt_lines.append(target_output + "\n\n")
                        prompt_lines.append(f"[Random {intervention_mode}]:\n")
                        prompt_lines.append(random_output + "\n\n")
                        entry["generations"].append(
                            {
                                "generation": generation,
                                "intervention_type": intervention_mode,
                                "target_indices": list(intervention_indices),
                                "target_weights": list(intervention_feature_weights or []),
                                "weighted_intervention_values": list(weighted_intervention_values or []),
                                "random_indices": list(random_indices),
                                "target_output": target_output,
                                "random_output": random_output,
                            }
                        )
                        print(
                            f"Prompt {entry['prompt_index']}, generation {generation} completed "
                            f"and buffered for {output_path}."
                        )

        experiment_results["prompt_results"] = prompt_entries

        with open(output_path, "a", encoding="utf-8") as out_file:
            for entry in prompt_entries:
                out_file.write("".join(prompt_lines_by_idx[entry["prompt_index"]]))
                out_file.flush()

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


class InterventionJudgeResult(BaseModel):
    target_control_success: bool
    grammar_preserved: bool
    meaning_preserved: bool
    target_control_reason: str
    grammar_reason: str
    meaning_reason: str


JUDGE_SYSTEM_PROMPT = (
    "You are a strict judge for intervention outputs on English modality or supposition control.\n"
    "Evaluate the output against exactly three criteria:\n"
    "1. target_control_success: Whether the requested ablation or enhancement of the target modality succeeded.\n"
    "2. grammar_preserved: Whether the output remains grammatical and coherent English.\n"
    "3. meaning_preserved: Whether the non-target meaning is preserved aside from the intended modality change.\n"
    "Be strict. Added content, semantic drift, contradiction, or malformed English should be penalized.\n"
    "Return only JSON matching the schema."
)


def _strip_prompt_prefix(prompt: str, output_text: str) -> str:
    normalized_prompt = prompt.strip()
    normalized_output = output_text.strip()
    if normalized_output.startswith(normalized_prompt):
        stripped = normalized_output[len(normalized_prompt):].strip()
        if stripped:
            return stripped
    return normalized_output


def _build_judge_user_prompt(
    target_modality: str,
    intervention_type: str,
    input_prompt: str,
    output_text: str,
) -> str:
    return (
        f"Target modality: {target_modality}\n"
        f"Requested intervention: {intervention_type}\n"
        "Interpretation of success:\n"
        "- ablation: the target modality should be weakened, removed, or made less explicit than in the input.\n"
        "- enhancement: the target modality should be strengthened, restored, or made more explicit than in the input.\n"
        "Judge the output text itself. If the model repeats the input and then adds continuation, judge the final output as produced.\n"
        f"Input text:\n{input_prompt}\n\n"
        f"Output text:\n{output_text}\n"
    )


def _call_judge_with_retries(
    client: OpenAI,
    target_modality: str,
    intervention_type: str,
    input_prompt: str,
    output_text: str,
    max_retries: int = 5,
    base_sleep: float = 1.0,
) -> InterventionJudgeResult:
    user_prompt = _build_judge_user_prompt(
        target_modality=target_modality,
        intervention_type=intervention_type,
        input_prompt=input_prompt,
        output_text=output_text,
    )
    last_err: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            resp = client.responses.parse(
                model=JUDGE_MODEL,
                input=[
                    {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                text_format=InterventionJudgeResult,
            )
            return resp.output_parsed
        except Exception as exc:
            last_err = exc
            sleep_s = min(base_sleep * (2 ** (attempt - 1)), 30.0) + 0.1 * attempt
            time.sleep(sleep_s)

    raise RuntimeError(f"Judge request failed after {max_retries} retries: {last_err}") from last_err


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


def _sample_random_intervention_indices(
    num_latents: int,
    num_indices: int,
    excluded_indices: List[int],
    rng: random.Random,
) -> List[int]:
    all_indices = set(range(num_latents))
    candidate_indices = sorted(all_indices - {int(idx) for idx in excluded_indices})
    if len(candidate_indices) < num_indices:
        raise ValueError(
            f"Not enough latent indices to sample {num_indices} random interventions."
        )
    return sorted(rng.sample(candidate_indices, num_indices))


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


def _get_intervention_plan(result: Dict[str, Any]) -> Tuple[List[int], float, List[float]]:
    intervention_features = result.get("intervention_features", [])
    feature_weights = result.get("intervention_feature_weights", [])
    if intervention_features:
        selected = {int(base_vec) for base_vec in intervention_features}
        ordered_entries = [
            item for item in feature_weights if int(item.get("base_vector")) in selected
        ]
        ordered_vectors = [int(item["base_vector"]) for item in ordered_entries]
        ordered_weights = [float(item.get("weight", 0.0)) for item in ordered_entries]
        if not ordered_vectors:
            ordered_vectors = [int(base_vec) for base_vec in intervention_features]
            ordered_weights = [0.0 for _ in ordered_vectors]
        score = max(
            (abs(float(item.get("weight", 0.0))) for item in feature_weights if int(item.get("base_vector")) in selected),
            default=0.0,
        )
        return ordered_vectors, float(score), ordered_weights

    top_features = result.get("top_features", [])
    if not top_features:
        return [], float("-inf"), []
    base_vec, score = top_features[0]
    return [int(curr_base_vec) for curr_base_vec, _ in top_features[:3]], float(score), []


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
        base_vectors, selection_score, feature_weights = _get_intervention_plan(result)
        if base_vectors:
            best["base_vectors"] = base_vectors
            best["selection_score"] = selection_score
            best["feature_weights"] = feature_weights
        return best

    layer_results = crosslayer_results.get("base_results", {}).get("layer_results", {})
    for layer_str, result in layer_results.items():
        base_vectors, selection_score, feature_weights = _get_intervention_plan(result)
        if not base_vectors:
            continue
        return {
            "base_vectors": base_vectors,
            "layer_idx": int(layer_str),
            "frc": float(selection_score),
            "feature_weights": feature_weights,
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
        base_vectors, selection_score, feature_weights = _get_intervention_plan(result)
        if not base_vectors:
            continue
        candidates.append(
            {
                "base_vectors": base_vectors,
                "layer_idx": int(layer_idx),
                "frc": float(selection_score),
                "feature_weights": feature_weights,
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


def _save_json(path: str, payload: Dict[str, Any]) -> str:
    output_dir = os.path.dirname(path) or "."
    os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return path


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


def _load_results_json(output_path: str) -> Dict[str, Any]:
    results_path = _get_result_paths(output_path)["results_json"]
    with open(results_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _judge_output_path(output_path: str) -> str:
    return output_path.replace(".txt", "_judge.json")


def _slugify_alpha(alpha: float) -> str:
    return str(alpha).replace("-", "neg_").replace(".", "p")


def _candidate_slug(candidate: Dict[str, Any]) -> str:
    base_vectors = "-".join(map(str, candidate.get("base_vectors", []))) or "none"
    return f"layer{candidate['layer_idx']}_bv{base_vectors}"


def _requires_alpha_tuning(candidate: Dict[str, Any]) -> bool:
    return bool(candidate.get("feature_weights"))


def _validate_alpha_tuning_args(args: argparse.Namespace) -> None:
    if not args.output_dir:
        raise ValueError("--output-dir is required for automatic alpha tuning.")
    if not args.dev_prompt_file or not args.test_prompt_file:
        raise ValueError(
            "--dev-prompt-file and --test-prompt-file are required for automatic alpha tuning."
        )
    if not args.target_modality:
        raise ValueError("--target-modality is required for automatic alpha tuning.")


def _run_saved_results_judge(
    output_path: str,
    target_modality: str,
    max_workers: int,
) -> Dict[str, Any]:
    experiment_results = _load_results_json(output_path)
    judge_results = _evaluate_intervention_results_with_judge(
        experiment_results=experiment_results,
        target_modality=target_modality,
        max_workers=max_workers,
    )
    judge_path = _save_json(_judge_output_path(output_path), judge_results)
    return {
        "judge_path": judge_path,
        "judge_summary": judge_results["summary"],
    }


def _evaluate_intervention_results_with_judge(
    experiment_results: Dict[str, Any],
    target_modality: str,
    max_workers: int = 32,
) -> Dict[str, Any]:
    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError("OPENAI_API_KEY is not set.")

    tasks = []
    for prompt_entry in experiment_results.get("prompt_results", []):
        for generation_entry in prompt_entry.get("generations", []):
            target_output = generation_entry.get("target_output", "")
            input_prompt = prompt_entry.get("input_prompt", "")
            tasks.append(
                {
                    "prompt_index": int(prompt_entry.get("prompt_index", -1)),
                    "generation": int(generation_entry.get("generation", -1)),
                    "intervention_type": generation_entry.get(
                        "intervention_type",
                        prompt_entry.get("intervention_type", "unknown"),
                    ),
                    "input_prompt": input_prompt,
                    "output_text": target_output,
                    "continuation_only_text": _strip_prompt_prefix(input_prompt, target_output),
                }
            )

    evaluations: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []

    def _run_single_judge(task: Dict[str, Any]) -> Dict[str, Any]:
        client = OpenAI()
        verdict = _call_judge_with_retries(
            client=client,
            target_modality=target_modality,
            intervention_type=task["intervention_type"],
            input_prompt=task["input_prompt"],
            output_text=task["output_text"],
        )
        return {
            **task,
            "judge": verdict.model_dump(),
            "all_criteria_pass": bool(
                verdict.target_control_success
                and verdict.grammar_preserved
                and verdict.meaning_preserved
            ),
        }

    with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as executor:
        futures = {executor.submit(_run_single_judge, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            try:
                evaluations.append(future.result())
            except Exception as exc:
                errors.append(
                    {
                        "prompt_index": task["prompt_index"],
                        "generation": task["generation"],
                        "intervention_type": task["intervention_type"],
                        "error": str(exc),
                    }
                )

    evaluations.sort(key=lambda item: (item["prompt_index"], item["generation"]))

    summary = {
        "target_modality": target_modality,
        "num_evaluated": len(evaluations),
        "num_errors": len(errors),
        "all_criteria_pass_count": sum(
            1 for item in evaluations if item["all_criteria_pass"]
        ),
        "target_control_success_count": sum(
            1 for item in evaluations if item["judge"]["target_control_success"]
        ),
        "grammar_preserved_count": sum(
            1 for item in evaluations if item["judge"]["grammar_preserved"]
        ),
        "meaning_preserved_count": sum(
            1 for item in evaluations if item["judge"]["meaning_preserved"]
        ),
        "by_intervention_type": {},
    }

    for intervention_type in ("ablation", "enhancement"):
        subset = [
            item for item in evaluations if item["intervention_type"] == intervention_type
        ]
        summary["by_intervention_type"][intervention_type] = {
            "count": len(subset),
            "all_criteria_pass_count": sum(
                1 for item in subset if item["all_criteria_pass"]
            ),
            "target_control_success_count": sum(
                1 for item in subset if item["judge"]["target_control_success"]
            ),
            "grammar_preserved_count": sum(
                1 for item in subset if item["judge"]["grammar_preserved"]
            ),
            "meaning_preserved_count": sum(
                1 for item in subset if item["judge"]["meaning_preserved"]
            ),
        }

    return {
        "experiment_name": experiment_results.get("experiment_name"),
        "layer_idx": experiment_results.get("layer_idx"),
        "alpha": experiment_results.get("alpha"),
        "target_intervention_indices": experiment_results.get(
            "target_intervention_indices", []
        ),
        "target_intervention_weights": experiment_results.get(
            "intervention_feature_weights", []
        ),
        "summary": summary,
        "evaluations": evaluations,
        "errors": errors,
    }


def _run_alpha_sweep_and_test(
    args: argparse.Namespace,
    candidate: Dict[str, Any],
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    _validate_alpha_tuning_args(args)
    if not candidate.get("feature_weights"):
        raise ValueError(
            "Selected intervention candidate does not contain feature-selection weights. "
            "Regenerate crosslayer outputs before running alpha tuning."
        )

    candidate_output_dir = output_dir or os.path.join(args.output_dir, _candidate_slug(candidate))
    os.makedirs(candidate_output_dir, exist_ok=True)
    with open(args.dev_prompt_file, "r", encoding="utf-8") as f:
        dev_prompt_text = f.read()
    with open(args.test_prompt_file, "r", encoding="utf-8") as f:
        test_prompt_text = f.read()

    alpha_results = []
    best_alpha_result = None

    for alpha in args.alpha_values:
        alpha_slug = _slugify_alpha(alpha)
        output_path = os.path.join(candidate_output_dir, f"dev_alpha_{alpha_slug}.txt")
        print(f"Running dev alpha sweep: alpha={alpha}")
        selection_meta = _run_single_candidate(
            args=args,
            candidate=candidate,
            prompt_text=dev_prompt_text,
            output_path=output_path,
            experiment_name=f"dev_alpha_{alpha_slug}",
            alpha=alpha,
        )
        experiment_results = _load_results_json(output_path)
        judge_results = _evaluate_intervention_results_with_judge(
            experiment_results=experiment_results,
            target_modality=args.target_modality,
            max_workers=args.judge_max_workers,
        )
        judge_path = _save_json(_judge_output_path(output_path), judge_results)
        alpha_result = {
            "alpha": float(alpha),
            "output_path": output_path,
            "judge_path": judge_path,
            "selection_meta": selection_meta,
            "summary": judge_results["summary"],
        }
        alpha_results.append(alpha_result)

        if best_alpha_result is None:
            best_alpha_result = alpha_result
            continue
        current_pass = alpha_result["summary"]["all_criteria_pass_count"]
        best_pass = best_alpha_result["summary"]["all_criteria_pass_count"]
        if current_pass > best_pass or (
            current_pass == best_pass and float(alpha) < float(best_alpha_result["alpha"])
        ):
            best_alpha_result = alpha_result

    assert best_alpha_result is not None
    best_alpha = float(best_alpha_result["alpha"])
    summary_path = os.path.join(candidate_output_dir, "alpha_sweep_summary.json")
    alpha_summary = {
        "target_modality": args.target_modality,
        "candidate": {
            "cross_json": candidate.get("cross_json", args.crosslayer_json),
            "layer_idx": candidate["layer_idx"],
            "base_vectors": candidate["base_vectors"],
            "feature_weights": candidate.get("feature_weights", []),
        },
        "selected_alpha": best_alpha,
        "alpha_results": alpha_results,
    }
    _save_json(summary_path, alpha_summary)

    test_output_path = os.path.join(
        candidate_output_dir, f"test_selected_alpha_{_slugify_alpha(best_alpha)}.txt"
    )
    print(f"Running test evaluation with selected alpha={best_alpha}")
    test_selection = _run_single_candidate(
        args=args,
        candidate=candidate,
        prompt_text=test_prompt_text,
        output_path=test_output_path,
        experiment_name=f"test_alpha_{_slugify_alpha(best_alpha)}",
        alpha=best_alpha,
    )
    test_results = _load_results_json(test_output_path)
    test_judge = _evaluate_intervention_results_with_judge(
        experiment_results=test_results,
        target_modality=args.target_modality,
        max_workers=args.judge_max_workers,
    )
    test_judge_path = _save_json(_judge_output_path(test_output_path), test_judge)

    final_summary = {
        "target_modality": args.target_modality,
        "candidate": {
            "cross_json": candidate.get("cross_json", args.crosslayer_json),
            "layer_idx": candidate["layer_idx"],
            "base_vectors": candidate["base_vectors"],
            "feature_weights": candidate.get("feature_weights", []),
        },
        "selected_alpha": best_alpha,
        "output_dir": candidate_output_dir,
        "dev_summary_path": summary_path,
        "test_output_path": test_output_path,
        "test_selection": test_selection,
        "test_judge_path": test_judge_path,
        "test_summary": test_judge["summary"],
    }
    _save_json(
        os.path.join(candidate_output_dir, "test_evaluation_summary.json"),
        final_summary,
    )
    return final_summary


def _run_single_candidate(
    args: argparse.Namespace,
    candidate: Dict[str, Any],
    prompt_text: str,
    output_path: str,
    experiment_name: Optional[str] = None,
    alpha: Optional[float] = None,
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
            "selected_feature_weights": candidate.get("feature_weights", []),
            "selected_frc": candidate["frc"],
            "sae_path": args.sae_path_template.format(candidate["layer_idx"]),
            "alpha": alpha,
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
        "selected_feature_weights": candidate.get("feature_weights", []),
        "selected_frc": candidate["frc"],
        "sae_path": sae_path,
        "input_prompts": input_prompts,
        "alpha": alpha,
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
            random_seed=args.random_seed,
            batch_size=args.batch_size,
            alpha=alpha,
            intervention_feature_weights=candidate.get("feature_weights") or None,
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
    parser.add_argument(
        "--dev-prompt-file",
        default=None,
        help="Dev prompt text file used for alpha tuning with the LLM judge.",
    )
    parser.add_argument(
        "--test-prompt-file",
        default=None,
        help="Test prompt text file used for final evaluation with the selected alpha.",
    )
    prompt_group = parser.add_mutually_exclusive_group(required=False)
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
    parser.add_argument(
        "--random-seed",
        type=int,
        default=0,
        help="Seed used to sample comparison intervention indices.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of prompts to generate together for the same intervention setting.",
    )
    parser.add_argument(
        "--target-modality",
        default=None,
        help="Target modality string, required when running alpha tuning / judge evaluation.",
    )
    parser.add_argument(
        "--alpha-values",
        default="0.25,0.5,1.0,2.0",
        help="Comma-separated alpha values for weighted intervention tuning.",
    )
    parser.add_argument(
        "--judge-max-workers",
        type=int,
        default=int(os.getenv("MAX_WORKERS", "32")),
        help="Max parallel judge requests for gpt-5-mini evaluation.",
    )

    args = parser.parse_args()
    if args.selection_mode == "per-layer" and not args.output_dir:
        raise ValueError("--output-dir is required when --selection-mode=per-layer.")
    args.alpha_values = [
        float(value.strip()) for value in args.alpha_values.split(",") if value.strip()
    ]

    crosslayer_entries = _load_crosslayer_jsons(args.crosslayer_json)

    if args.selection_mode == "best":
        best = _select_best_intervention_candidate_from_many(crosslayer_entries)
        if _requires_alpha_tuning(best):
            if not args.output_dir:
                raise ValueError(
                    "--output-dir is required when the selected candidate has feature weights."
                )
            _run_alpha_sweep_and_test(
                args=args,
                candidate=best,
                output_dir=os.path.join(args.output_dir, _candidate_slug(best)),
            )
            return 0
        if not args.output_path:
            raise ValueError("--output-path is required when --selection-mode=best.")
        if not args.prompt and not args.prompt_file:
            raise ValueError("Specify --prompt or --prompt-file.")
        prompt_text = args.prompt
        if args.prompt_file:
            with open(args.prompt_file, "r", encoding="utf-8") as f:
                prompt_text = f.read()
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
    requires_prompt = any(not _requires_alpha_tuning(candidate) for candidate in candidates)
    prompt_text = args.prompt
    if requires_prompt:
        if not args.prompt and not args.prompt_file:
            raise ValueError("Specify --prompt or --prompt-file.")
        if args.prompt_file:
            with open(args.prompt_file, "r", encoding="utf-8") as f:
                prompt_text = f.read()

    for candidate in candidates:
        if _requires_alpha_tuning(candidate):
            _run_alpha_sweep_and_test(
                args=args,
                candidate=candidate,
                output_dir=os.path.join(args.output_dir, _candidate_slug(candidate)),
            )
            continue

        output_path = os.path.join(
            args.output_dir, f"layer{candidate['layer_idx']}_intervention.txt"
        )
        experiment_name = args.experiment_name or (
            f"intervention_layer{candidate['layer_idx']}_bv{'-'.join(map(str, candidate['base_vectors']))}"
        )
        selection_meta = _run_single_candidate(
            args=args,
            candidate=candidate,
            prompt_text=prompt_text,
            output_path=output_path,
            experiment_name=experiment_name,
        )
        if args.target_modality:
            judge_meta = _run_saved_results_judge(
                output_path=output_path,
                target_modality=args.target_modality,
                max_workers=args.judge_max_workers,
            )
            per_layer_summary = {
                "target_modality": args.target_modality,
                "candidate": {
                    "cross_json": candidate.get("cross_json", args.crosslayer_json),
                    "layer_idx": candidate["layer_idx"],
                    "base_vectors": candidate["base_vectors"],
                    "feature_weights": candidate.get("feature_weights", []),
                },
                "output_path": output_path,
                "selection_meta": selection_meta,
                "test_judge_path": judge_meta["judge_path"],
                "test_summary": judge_meta["judge_summary"],
            }
            _save_json(
                os.path.join(
                    args.output_dir,
                    f"{_candidate_slug(candidate)}_test_evaluation_summary.json",
                ),
                per_layer_summary,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
