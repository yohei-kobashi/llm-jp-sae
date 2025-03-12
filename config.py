from dataclasses import dataclass, field

import torch
from transformers import PretrainedConfig


@dataclass
class UsrConfig:
    raw_data_dir: str = ""
    tokenized_data_dir: str = ""
    llmjp_model_dir: str = ""
    sae_save_dir: str = ""


@dataclass
class DataConfig:
    seq_len: int = 64
    token_num: int = 165_000_000
    data_pths: dict = field(
        default_factory=lambda: {
            "ja_wiki": [f"ja/ja_wiki/train_{str(i)}.jsonl" for i in range(14)],
            "en_wiki": [f"en/en_wiki/train_{str(i)}.jsonl" for i in range(67)],
        }
    )
    batch_size_tokenizer: int = 5000
    train_val_test_ratio: list = field(default_factory=lambda: [0.8, 0.1, 0.1])


class SaeConfig(PretrainedConfig):
    def __init__(
        self,
        d_in: int = 2048,
        expansion_factor: int = 16,
        k: int = 32,
        torch_dtype: torch.dtype = torch.bfloat16,
        **kwargs,
    ):
        super().__init__(**kwargs, torch_dtype=torch_dtype)
        self.d_in = d_in
        self.expansion_factor = expansion_factor
        self.k = k


@dataclass
class TrainConfig:
    lr_warmup_steps: int = 1000
    batch_size: int = 512
    inf_bs_expansion: int = 2
    logging_step: int = 50


@dataclass
class EvalConfig:
    num_examples: int = 50
    act_threshold_p: float = 0.7


def return_save_dir(root_dir, layer, n_d, k, nl, ckpt, lr):
    save_dir = root_dir
    save_dir += f"/layer_{layer}"
    save_dir += f"/n_d_{n_d}"
    save_dir += f"/k_{k}"
    save_dir += f"/nl_{nl}"
    save_dir += f"/ckpt_{str(ckpt).zfill(7)}"
    save_dir += f"/lr_{lr}"
    return save_dir
