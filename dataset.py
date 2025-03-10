from torch.utils.data import Dataset
import torch

class CustomWikiDataset(Dataset):
    def __init__(self, data_path, output_lang=False):
        self.data = torch.load(data_path, weights_only=True)
        self.all_len = self.data.shape[0]
        self.output_lang = output_lang

    def __len__(self):
        return self.all_len

    def __getitem__(self, idx):
        if self.output_lang:
            return self.data[idx][-1], self.data[idx][:-1]
        else:
            return self.data[idx]
