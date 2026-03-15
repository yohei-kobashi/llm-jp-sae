"""
OpenSAE-style intervention utilities adapted for this repository's SAE.

This file ports intervention-related classes from OpenSAE's
`transformer_with_sae.py` and adapts them to `model.SparseAutoEncoder`
checkpoints produced by this repository.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import transformers
from transformers import PretrainedConfig

from config import SaeConfig
from model import SparseAutoEncoder, normalize_activation


@dataclass
class TrainSaeEncoderOutput:
    sparse_feature_indices: torch.Tensor
    sparse_feature_activations: torch.Tensor


def extend_encoder_output(
    previous: Optional[TrainSaeEncoderOutput],
    current: TrainSaeEncoderOutput,
) -> TrainSaeEncoderOutput:
    if previous is None:
        return current
    return TrainSaeEncoderOutput(
        sparse_feature_indices=torch.cat(
            [previous.sparse_feature_indices, current.sparse_feature_indices], dim=0
        ),
        sparse_feature_activations=torch.cat(
            [previous.sparse_feature_activations, current.sparse_feature_activations], dim=0
        ),
    )


class InterventionConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.intervention = kwargs.pop("intervention", False)
        self.intervention_mode = kwargs.pop("intervention_mode", "set")  # set, multiply, add
        assert self.intervention_mode in ["set", "multiply", "add"], (
            "intervention_mode must be one of `set`, `multiply`, or `add`."
        )

        self.intervention_indices = kwargs.pop("intervention_indices", None)
        if self.intervention:
            assert self.intervention_indices is not None, (
                "intervention indices are not provided when set intervention to True"
            )
        self.intervention_value = kwargs.pop("intervention_value", 0.0)
        self.intervention_weights = kwargs.pop("intervention_weights", None)
        if self.intervention_weights is not None and self.intervention_indices is not None:
            assert len(self.intervention_weights) == len(self.intervention_indices), (
                "intervention_weights must have the same length as intervention_indices."
            )
        self.prompt_only = kwargs.pop("prompt_only", False)


class TransformerWithSae(torch.nn.Module):
    """
    OpenSAE-like wrapper for intervention with train.py SAE checkpoints.

    Differences from OpenSAE:
    - Uses `SparseAutoEncoder` from this repo.
    - Uses explicit `layer_idx` instead of SAE hookpoint config.
    - Reconstructs hidden states from sparse latents via `W_dec`/`b_dec`.
    """

    def __init__(
        self,
        transformer: transformers.PreTrainedModel | Path | str,
        sae: SparseAutoEncoder | Path | str,
        device: str | torch.device = "cpu",
        intervention_config: InterventionConfig | None = None,
        layer_idx: int = 0,
        normalization: str = "Scalar",
        k: int = 32,
    ):
        super().__init__()

        if isinstance(transformer, (Path, str)):
            self.transformer = transformers.AutoModelForCausalLM.from_pretrained(transformer)
        else:
            self.transformer = transformer

        if isinstance(sae, (Path, str)):
            self.sae = self._load_train_sae_from_checkpoint(str(sae), device, k)
        else:
            self.sae = sae

        self.device = device
        self.transformer.to(self.device)
        self.sae.to(self.device)
        self.sae.eval()

        self.layer_idx = int(layer_idx)
        self.normalization = normalization

        self.token_indices = None
        self.encoder_output: Optional[TrainSaeEncoderOutput] = None
        self.saved_features: Optional[TrainSaeEncoderOutput] = None
        self.prefilling_stage = True

        self.forward_hook_handle = {}
        self._register_input_hook()

        self.intervention_config = intervention_config or InterventionConfig()

    @staticmethod
    def _load_train_sae_from_checkpoint(
        ckpt_path: str,
        device: str | torch.device,
        k: int,
    ) -> SparseAutoEncoder:
        state = torch.load(ckpt_path, map_location=device)
        if "b_dec" in state:
            d_in = int(state["b_dec"].shape[0])
        elif "encoder.weight" in state:
            d_in = int(state["encoder.weight"].shape[1])
        else:
            raise KeyError(f"Cannot infer SAE input width from checkpoint: {ckpt_path}")

        if "encoder.weight" in state:
            num_latents = int(state["encoder.weight"].shape[0])
        elif "W_dec" in state:
            num_latents = int(state["W_dec"].shape[0])
        else:
            raise KeyError(f"Cannot infer SAE latent width from checkpoint: {ckpt_path}")

        if num_latents % d_in != 0:
            raise ValueError(
                f"Invalid SAE shape in {ckpt_path}: num_latents={num_latents}, d_in={d_in}"
            )

        cfg = SaeConfig(d_in=d_in, expansion_factor=num_latents // d_in, k=int(k))
        sae_model = SparseAutoEncoder(cfg).to(device)
        sae_model.load_state_dict(state)
        return sae_model

    def clear_intermediates(self):
        self.token_indices = None
        self.encoder_output = None
        self.saved_features = None
        self.prefilling_stage = True

    def _resolve_hook_module(self):
        core = getattr(self.transformer, "model", None) or getattr(
            self.transformer, "transformer", None
        )
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
        if self.layer_idx == 0:
            return embed_tokens
        if self.layer_idx - 1 < 0 or self.layer_idx - 1 >= len(layers):
            raise IndexError(
                f"layer_idx={self.layer_idx} out of range for model with {len(layers)} layers."
            )
        return layers[self.layer_idx - 1]

    def _register_input_hook(self):
        module = self._resolve_hook_module()
        hook_name = f"layer_{self.layer_idx}"
        self.forward_hook_handle[hook_name] = module.register_forward_hook(
            self._input_hook_fn
        )

    def _input_hook_fn(self, module, input_, output):
        if self.intervention_config.prompt_only and not self.prefilling_stage:
            return

        output_tensor = output[0] if isinstance(output, tuple) else output
        bsz, seq_len, hidden_size = output_tensor.size()

        if self.token_indices is None:
            sae_input = output_tensor
        else:
            sae_input = output_tensor[self.token_indices]
        sae_input = sae_input.reshape(-1, hidden_size)

        sae_dtype = self.sae.encoder.weight.dtype
        sae_input = sae_input.to(dtype=sae_dtype)
        sae_input = normalize_activation(sae_input, self.normalization)

        with torch.no_grad():
            sae_forward = self.sae(sae_input)
        self.encoder_output = TrainSaeEncoderOutput(
            sparse_feature_indices=sae_forward.latent_indices,
            sparse_feature_activations=sae_forward.latent_acts,
        )
        self.saved_features = extend_encoder_output(self.saved_features, self.encoder_output)

        if self.intervention_config.intervention:
            self._apply_intervention()

        return self._output_hook_fn(output, bsz, seq_len, hidden_size)

    def _output_hook_fn(self, output, bsz: int, seq_len: int, hidden_size: int):
        if self.intervention_config.prompt_only and not self.prefilling_stage:
            return
        assert self.encoder_output is not None, "encoder_output is None"

        output_tensor = output[0] if isinstance(output, tuple) else output
        output_dtype = output_tensor.dtype

        n_tokens, k = self.encoder_output.sparse_feature_indices.shape
        num_latents = int(self.sae.W_dec.shape[0])
        dense = torch.zeros(
            (n_tokens, num_latents),
            device=self.encoder_output.sparse_feature_activations.device,
            dtype=self.encoder_output.sparse_feature_activations.dtype,
        )
        dense.scatter_(
            dim=-1,
            index=self.encoder_output.sparse_feature_indices,
            src=self.encoder_output.sparse_feature_activations,
        )

        sae_output = dense @ self.sae.W_dec + self.sae.b_dec

        if self.token_indices is not None:
            reconstructed_output = output_tensor
            reconstructed_output[self.token_indices] = sae_output.reshape(-1, hidden_size)
        else:
            reconstructed_output = sae_output.reshape(bsz, seq_len, hidden_size)

        reconstructed_output = reconstructed_output.to(output_dtype)
        self.prefilling_stage = False

        if isinstance(output, tuple):
            return (reconstructed_output,) + output[1:]
        return reconstructed_output

    def _apply_intervention(self):
        mode = self.intervention_config.intervention_mode
        if mode == "multiply":
            self._apply_intervention_multiply()
        elif mode in ["add", "set"]:
            self._apply_intervention_add_or_set()

    def _apply_intervention_multiply(self):
        for position, intervention_index in enumerate(self.intervention_config.intervention_indices):
            intervention_value = self._get_intervention_value(position)
            mask = self.encoder_output.sparse_feature_indices == intervention_index
            self.encoder_output.sparse_feature_activations[mask] *= (
                intervention_value
            )

    def _apply_intervention_add_or_set(self):
        for position, intervention_index in enumerate(self.intervention_config.intervention_indices):
            intervention_value = self._get_intervention_value(position)
            mask = self.encoder_output.sparse_feature_indices == intervention_index
            is_ind_activated = torch.any(mask, dim=-1)

            if self.intervention_config.intervention_mode == "add":
                self.encoder_output.sparse_feature_activations[mask] += (
                    intervention_value
                )
            elif self.intervention_config.intervention_mode == "set":
                self.encoder_output.sparse_feature_activations[mask] = (
                    intervention_value
                )

            if not torch.all(is_ind_activated):
                min_val, min_ind = torch.min(
                    self.encoder_output.sparse_feature_activations, dim=-1
                )
                token_select = torch.arange(
                    0,
                    len(min_ind),
                    dtype=torch.long,
                    device=self.encoder_output.sparse_feature_activations.device,
                )

                set_val = min_val.clone()
                set_val[~is_ind_activated] = intervention_value

                set_ind = self.encoder_output.sparse_feature_indices[token_select, min_ind]
                set_ind[~is_ind_activated] = intervention_index

                self.encoder_output.sparse_feature_activations[token_select, min_ind] = set_val
                self.encoder_output.sparse_feature_indices[token_select, min_ind] = set_ind

    def _get_intervention_value(self, position: int) -> float:
        if self.intervention_config.intervention_weights is not None:
            return float(self.intervention_config.intervention_weights[position])
        return float(self.intervention_config.intervention_value)

    def update_intervention_config(self, intervention_config: InterventionConfig):
        self.intervention_config = intervention_config

    def forward(self, return_features=False, intervention_config=None, *inputs, **kwargs):
        self.clear_intermediates()
        if intervention_config is not None:
            self.update_intervention_config(intervention_config)
        forward_output = self.transformer(*inputs, **kwargs)
        if return_features:
            return self.encoder_output, forward_output
        return forward_output

    def generate(self, return_features=False, intervention_config=None, *inputs, **kwargs):
        self.clear_intermediates()
        if intervention_config is not None:
            self.update_intervention_config(intervention_config)
        generation = self.transformer.generate(*inputs, **kwargs)
        if return_features:
            return self.saved_features, generation
        return generation

    def remove_hooks(self):
        for handle in self.forward_hook_handle.values():
            try:
                handle.remove()
            except Exception:
                pass
        self.forward_hook_handle.clear()
