import torch
import torch.nn as nn
import torch.optim as optim
from dataset import MaskedCGMDataset
from model import MaskedCGMTransformer

# Load data
masked_data = np.load("data/processed/masked_windows_subset.npy")
mask_labels = np.load("data/processed/mask_labels_subset.npy")
dataset = MaskedCGMDataset(masked_data, mask_labels)
loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

# Model and training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MaskedCGMTransformer(embed_dim=64, num_heads=4, ff_dim=128, num_layers=4).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Training loop
for epoch in range(10):
    model.train()
    epoch_loss = 0
    for data, mask in loader:
        data, mask = data.to(device), mask.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        output = model(data)
        loss = criterion(output[mask], data[mask])  # Only calculate loss for masked values
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    print(f"Epoch {epoch + 1}, Loss: {epoch_loss / len(loader)}")

# Save final model
torch.save(model.state_dict(), "../models/baby_transformer.pt")
