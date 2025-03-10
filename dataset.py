from torch.utils.data import Dataset
import torch

class CustomWikiDataset(Dataset):
    def __init__(self, data_path):
        self.data = torch.load(data_path, weights_only=True)
        self.all_len = self.data.shape[0]

    def __len__(self):
        return self.all_len

    def __getitem__(self, idx):
        return self.data[idx]
