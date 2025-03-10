from dataclasses import dataclass

import torch


@dataclass
class UsrConfig:
    raw_data_dir: str = "llm-jp-corpus-v3"
    tokenized_data_dir: str = "llm-jp-corpus-v3/tokenized"
    llmjp_model_dir: str = "/model/inaba/llmjp_1.8B"
    model_save_dir: str = "trained_saes"


@dataclass
class DataConfig:
    seq_len: int = 64
    token_num: int = 165_000_000
    data_pths: dict = {
        "ja_wiki": [f"ja/ja_wiki/train_{str(i)}.jsonl" for i in range(14)],
        "en_wiki": [f"en/en_wiki/train_{str(i)}.jsonl" for i in range(67)],
    }
    batch_size_tokenizer: int = 5000
    train_val_test_ratio: list = [0.8, 0.1, 0.1]


@dataclass
class SaeConfig:
    d_in: int = 2048
    dtype: torch.dtype = torch.bfloat16
    n_d: int = 16
    k: int = 32


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
