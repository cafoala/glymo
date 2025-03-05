import torch
from model import TransformerModel
from dataset import MaskedCGMDataset
from torch.utils.data import DataLoader
import numpy as np

# Parameters
test_masked_file = "data/processed/masked_windows_test.npy"
test_labels_file = "data/processed/mask_labels_test.npy"
model_path = "models/baby_transformer_cgm.pth"
batch_size = 32
mask_token = -1

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load test dataset
test_dataset = MaskedCGMDataset(test_masked_file, test_labels_file, mask_token=mask_token)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Load the model
model = TransformerModel(embed_dim=64, num_heads=4, num_layers=2, dropout=0.1)
model.load_state_dict(torch.load(model_path))
model = model.to(device)
model.eval()

# Evaluation loop
all_preds, all_labels = [], []
with torch.no_grad():
    for batch in test_dataloader:
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(inputs)
        
        # Store results
        mask = labels != mask_token
        all_preds.append(outputs[mask].cpu().numpy())
        all_labels.append(labels[mask].cpu().numpy())

# Compute metrics
all_preds = np.concatenate(all_preds)
all_labels = np.concatenate(all_labels)
rmse = np.sqrt(np.mean((all_preds - all_labels) ** 2))
mae = np.mean(np.abs(all_preds - all_labels))

print(f"RMSE: {rmse:.4f}, MAE: {mae:.4f}")
