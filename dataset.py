from dataclasses import dataclass, field
from typing import List

import torch
from torch.utils.data import Dataset


class CustomWikiDataset(Dataset):
    def __init__(self, data_path):
        self.data = torch.load(data_path, weights_only=True)
        self.all_len = self.data.shape[0]

    def __len__(self):
        return self.all_len

    def __getitem__(self, idx):
        return self.data[idx]


@dataclass
class ActivationRecord:
    tokens: List[str]
    act_values: List[float]


@dataclass
class FeatureRecord:
    feature_id: int
    act_patterns: List[ActivationRecord] = field(default_factory=list)
