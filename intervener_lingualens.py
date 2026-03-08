"""
Model intervention module for train.py SAE checkpoints.

This module mirrors LinguaLens.lingualens.intervener.Intervener but uses
OpenSAE-style intervention utilities implemented in local `util.py`.
"""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional

import torch

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
        self.tokenizer = setup_tokenizer(model_path)

        self.model: Optional[TransformerWithSae] = None
        self._intervention_config = InterventionConfig(
            intervention=False,
            intervention_indices=[],
            intervention_value=0.0,
        )

        self._load_models()

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
        print(f"Loading transformer model from {self.model_path}")

        self.model = TransformerWithSae(
            transformer=self.model_path,
            sae=self.sae_path,
            device=self.device,
            intervention_config=self._intervention_config,
            layer_idx=self.layer_idx,
            normalization=self.normalization,
            k=self.k,
        )

        if self.torch_dtype is not None:
            try:
                self.model.transformer.to(dtype=self.torch_dtype)
            except Exception:
                pass

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
        experiment_results = {
            "experiment_name": experiment_name or "intervention_experiment",
            "input_prompt": input_prompt,
            "intervention_indices": intervention_indices,
            "num_generations": num_generations,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "layer_idx": self.layer_idx,
            "conditions": {"ablation": [], "enhancement": [], "control": []},
        }

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as out_file:
            out_file.write(f"Experiment: {experiment_results['experiment_name']}\n\n")
            out_file.write(f"[INPUT PROMPT]: {input_prompt}\n\n")
            out_file.write(f"[INTERVENTION INDICES]: {intervention_indices}\n\n")
            out_file.write(f"[LAYER IDX]: {self.layer_idx}\n\n")

        progress = ProgressLogger(num_generations * 3, "Running intervention experiment")
        for generation in range(1, num_generations + 1):
            print(f"Starting generation {generation}...")
            inputs = self.tokenizer(input_prompt, return_tensors="pt").to(self.device)

            with open(output_path, "a", encoding="utf-8") as out_file:
                out_file.write(f"--- Generation {generation} ---\n")

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
                out_file.write("[Ablation (set value=0)]:\n")
                out_file.write(ablation_output + "\n\n")
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
                out_file.write("[Enhancement (set value=10)]:\n")
                out_file.write(enhancement_output + "\n\n")
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
                out_file.write("[Control (multiply value=1)]:\n")
                out_file.write(control_output + "\n\n")
                progress.update()

            experiment_results["conditions"]["ablation"].append(ablation_output)
            experiment_results["conditions"]["enhancement"].append(enhancement_output)
            experiment_results["conditions"]["control"].append(control_output)
            print(f"Generation {generation} completed and written to {output_path}.")

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
