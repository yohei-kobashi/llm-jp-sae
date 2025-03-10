from typing import NamedTuple

import torch
from torch import Tensor, nn


class SimpleHook:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn, with_kwargs=True)

    def hook_fn(self, module, args, kwargs, output):
        self.args = args
        self.kwargs = kwargs
        self.output = output


class EncoderOutput(NamedTuple):
    top_acts: Tensor
    top_indices: Tensor


class ForwardOutput(NamedTuple):
    sae_out: Tensor
    latent_acts: Tensor
    latent_indices: Tensor
    loss: Tensor


class SparseAutoEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.d_in = cfg.d_in
        self.num_latents = self.d_in * cfg.expansion_factor
        self.dtype = cfg.dtype

        self.encoder = nn.Linear(self.d_in, self.num_latents, dtype=self.dtype)
        self.encoder.bias.data.zero_()

        self.W_dec = nn.Parameter(self.encoder.weight.data.clone())
        if self.cfg.normalize_decoder:
            self.set_decoder_norm_to_unit_norm()

        self.b_dec = nn.Parameter(torch.zeros(self.d_in, dtype=self.dtype))

    def select_topk(self, latents: Tensor) -> EncoderOutput:
        """Select the top-k latents."""
        return EncoderOutput(*latents.topk(self.cfg.k, sorted=False))

    def forward(self, x: Tensor) -> ForwardOutput:
        sae_in = x.to(self.dtype) - self.b_dec

        latents = self.encoder(sae_in)
        latents = nn.functional.relu(latents)
        top_acts, top_indices = self.select_topk(latents)
        buf = top_acts.new_zeros(top_acts.shape[:-1] + (self.W_dec.shape[-2],))
        acts = buf.scatter_(dim=-1, index=top_indices, src=top_acts)

        sae_out = acts @ self.W_dec + self.b_dec

        # compute loss
        e = sae_out - x
        l2_loss = e.pow(2).sum()
        total_variance = (x - x.mean(0)).pow(2).sum()
        loss = l2_loss / total_variance

        return ForwardOutput(
            sae_out,
            top_acts,
            top_indices,
            loss,
        )

    @torch.no_grad()
    def set_decoder_norm_to_unit_norm(self):
        assert self.W_dec is not None, "Decoder weight was not initialized."

        eps = torch.finfo(self.W_dec.dtype).eps
        norm = torch.norm(self.W_dec.data, dim=1, keepdim=True)
        self.W_dec.data /= norm + eps
