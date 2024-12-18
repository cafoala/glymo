import torch
from torch.utils.data import Dataset

class MaskedCGMDataset(Dataset):
    def __init__(self, masked_data, mask_labels):
        self.masked_data = masked_data
        self.mask_labels = mask_labels

    def __len__(self):
        return len(self.masked_data)

    def __getitem__(self, idx):
        return torch.tensor(self.masked_data[idx], dtype=torch.float32), torch.tensor(self.mask_labels[idx], dtype=torch.bool)
