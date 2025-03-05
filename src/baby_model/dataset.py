import numpy as np
import torch
from torch.utils.data import Dataset

class MaskedCGMDataset(Dataset):
    def __init__(self, masked_file, labels_file, mask_token=-1):
        self.masked_data = np.load(masked_file, allow_pickle=True)  # Shape: (num_samples, 3168)
        self.labels = np.load(labels_file, allow_pickle=True)  # Shape: (num_samples, 96)
        self.mask_token = mask_token

    def __len__(self):
        return len(self.masked_data)

    def __getitem__(self, idx):
        masked_window = self.masked_data[idx]  # Shape: (3168,)
        label = self.labels[idx]  # Shape: (96,)

        # ✅ Convert to tensors
        masked_window = torch.tensor(masked_window, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)

        # ✅ Reshape `masked_window` to (96, 33) but `label` to (96, 1)
        masked_window = masked_window.view(96, 33)
        label = label.view(96, 1)  # ✅ Fix label shape!

        return masked_window, label
