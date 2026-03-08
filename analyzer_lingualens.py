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
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM

from config import SaeConfig
from model import SimpleHook, SparseAutoEncoder, normalize_activation
from LinguaLens.lingualens.analyzer import LinguisticAnalyzer
from LinguaLens.lingualens.metrics import compute_layer_stats
from LinguaLens.lingualens.utils import validate_layer_indices


class TrainSaeLinguisticAnalyzer(LinguisticAnalyzer):
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

    def _analyze_single_layer(
        self, lines: List[str], layer_idx: int
    ) -> Dict[int, Dict[str, float]]:
        runtime = self._get_sae_model(layer_idx)
        model = self._load_base_model()
        sae: SparseAutoEncoder = runtime["sae"]
        hook: SimpleHook = runtime["hook"]

        structured_data: List[Dict[str, Any]] = []

        for start in range(0, len(lines), self.batch_size):
            batch_lines = lines[start : start + self.batch_size]
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
            attention_mask = attention_mask.to(self.device)

            with torch.no_grad():
                _ = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )

            acts = hook.output
            if isinstance(acts, tuple):
                acts = acts[0]
            if acts.dim() == 2:
                acts = acts.unsqueeze(0)
            acts = acts[:, 1:, :]
            token_mask = attention_mask[:, 1:]

            for i in range(input_ids.size(0)):
                valid_len = int(token_mask[i].sum().item())
                token_ids = input_ids[i, 1 : valid_len + 1]
                tokens = self.tokenizer.convert_ids_to_tokens(token_ids.tolist())
                tokens = [tok.replace("▁", " ") for tok in tokens]

                sentence_tokens: List[Dict[str, Any]] = []
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
                                token_activations.append(
                                    {
                                        "base_vector": int(base_vector),
                                        "activation": float(value),
                                    }
                                )
                        sentence_tokens.append(
                            {"token": token, "activations": token_activations}
                        )

                structured_data.append(
                    {
                        "sentence_id": start + i + 1,
                        "tokens": sentence_tokens,
                    }
                )

            hook.output = None

        return compute_layer_stats(structured_data)

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
