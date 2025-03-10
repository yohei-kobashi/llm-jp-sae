import json
import os
from typing import List

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from config import DataConfig, UsrConfig


def batch_tokenize(
    tokenizer: AutoTokenizer, texts: List[str], max_length: int, pad_id: int
) -> torch.Tensor:
    tokenized_texts = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )["input_ids"]

    # Remove sequences with padding tokens
    valid_mask = tokenized_texts.ne(pad_id).all(dim=1)
    return tokenized_texts[valid_mask]


def save_tensor(data: torch.Tensor, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(data, path)
    print(f"Saved data to {path}")


def mix_and_split_data(save_dir: str, token_num: int, ratios: List[float]) -> None:
    try:
        ja_tokenized = torch.load(os.path.join(save_dir, "tokenized_ja_wiki.pt"))
        en_tokenized = torch.load(os.path.join(save_dir, "tokenized_en_wiki.pt"))
    except FileNotFoundError as e:
        raise RuntimeError(f"Missing tokenized data: {e}")

    combined_data = torch.cat([ja_tokenized, en_tokenized], dim=0)
    shuffled_data = combined_data[torch.randperm(combined_data.size(0))]

    train_ratio = ratios[0]
    val_ratio = ratios[1]
    train_num = int(token_num * train_ratio)
    val_num = int(token_num * val_ratio)

    train_data = shuffled_data[:train_num]
    val_data = shuffled_data[train_num : train_num + val_num]
    test_data = shuffled_data[train_num + val_num : token_num]

    save_tensor(train_data, os.path.join(save_dir, "train_data.pt"))
    save_tensor(val_data, os.path.join(save_dir, "val_data.pt"))
    save_tensor(test_data, os.path.join(save_dir, "test_data.pt"))


def tokenize_and_save(
    data_type: str,
    file_paths: List[str],
    tokenizer: AutoTokenizer,
    max_length: int,
    pad_id: int,
    data_dir: str,
    save_dir: str,
    token_num: int,
    batch_size: int,
) -> None:
    total_tokens = 0
    tokenized_data = torch.zeros((token_num, max_length), dtype=torch.int32)

    for file_path in tqdm(file_paths, desc=f"Tokenizing {data_type}"):
        full_path = os.path.join(data_dir, file_path)
        if not os.path.exists(full_path):
            print(f"Warning: File {full_path} not found.")
            continue

        with open(full_path, "r", encoding="utf-8") as file:
            texts = [json.loads(line)["text"] for line in file]

        for start in range(0, len(texts), batch_size):
            end = min(start + batch_size, len(texts))
            batch_texts = texts[start:end]
            tokenized_batch = batch_tokenize(tokenizer, batch_texts, max_length, pad_id)

            num_tokens = tokenized_batch.size(0)
            if total_tokens + num_tokens > token_num:
                remaining_tokens = token_num - total_tokens
                tokenized_data[total_tokens:token_num] = tokenized_batch[
                    :remaining_tokens
                ]
                total_tokens = token_num
                break

            tokenized_data[total_tokens : total_tokens + num_tokens] = tokenized_batch
            total_tokens += num_tokens

        if total_tokens >= token_num:
            break

    save_tensor(
        tokenized_data[:total_tokens],
        os.path.join(save_dir, f"tokenized_{data_type}.pt"),
    )


def main():
    usr_cfg = UsrConfig()
    data_cfg = DataConfig()
    max_seq_len = data_cfg.seq_len + 1
    token_num = data_cfg.token_num
    data_paths = data_cfg.raw_data_pths
    batch_size = data_cfg.batch_size_tokenizer

    tokenizer = AutoTokenizer.from_pretrained("llm-jp/llm-jp-3-1.8b")

    # Check if tokenizer has a pad token
    if tokenizer.pad_token is None:
        raise ValueError("Tokenizer does not have a pad token defined.")

    pad_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

    token_num_each = token_num // 2

    for data_type, paths in data_paths.items():
        tokenize_and_save(
            data_type=data_type,
            file_paths=paths,
            tokenizer=tokenizer,
            max_length=max_seq_len,
            pad_id=pad_id,
            data_dir=usr_cfg.raw_data_dir,
            save_dir=usr_cfg.tokenized_data_dir,
            token_num=token_num_each,
            batch_size=batch_size,
        )

    mix_and_split_data(
        save_dir=usr_cfg.tokenized_data_dir,
        token_num=token_num,
        ratios=data_cfg.train_val_test_ratio,
    )

    print("Tokenization and preprocessing complete.")


if __name__ == "__main__":
    main()
