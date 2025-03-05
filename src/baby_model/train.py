import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model import TransformerModel  # Ensure this is updated with (batch_size, 96, 32)
from dataset import MaskedCGMDataset  # Ensure this correctly reshapes input
import numpy as np

# ✅ Hyperparameters
embed_dim = 32
num_heads = 4
num_layers = 2
dropout = 0.1
learning_rate = 1e-4
batch_size = 32
epochs = 10
mask_token = -1
ff_dim = 128  # Feedforward dim inside Transformer layer

# ✅ File paths
#masked_file = "data/processed/masked_windows_aleppo.npy"
#labels_file = "data/processed/mask_labels_aleppo.npy"

masked_file = "data/processed/masked_windows_lynch.npy"
labels_file = "data/processed/mask_labels_lynch.npy"

model_save_path = "models/baby_transformer_cgm.pth"

# ✅ Load dataset and verify shape
data = np.load(masked_file)
print("Final dataset shape:", data.shape)  # Should be (num_samples, 96, 32)

# ✅ Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Load dataset
dataset = MaskedCGMDataset(masked_file, labels_file, mask_token=mask_token)  # Ensure this loads correctly
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# ✅ Initialize the model
model = TransformerModel(embed_dim=embed_dim, num_heads=num_heads, ff_dim=ff_dim, num_layers=num_layers, dropout=dropout)
model = model.to(device)

# ✅ Verify first batch shape before training
for batch in dataloader:
    inputs, labels = batch
    print("Input shape before passing to model:", inputs.shape)  # Should be (batch_size, 96, 32)
    inputs, labels = inputs.to(device), labels.to(device)
    
    # Forward pass
    outputs = model(inputs)
    print("Output shape:", outputs.shape)  # Should match (batch_size, 96, 32)
    break  # Exit after one batch for debugging

# ✅ Loss function and optimizer
criterion = nn.MSELoss()  # Mean squared error for regression
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# ✅ Training loop
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for batch in dataloader:
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Forward pass
        outputs = model(inputs)

        # ✅ Mask should be of shape (batch_size, 96, 33)
        mask = labels != mask_token  # Mask tensor where labels are NOT masked

        # ✅ Expand mask to match output shape (batch_size, 96, 33)
        mask = mask.expand(-1, -1, 33)  # Keep batch and time dimensions, expand feature dimension

        # ✅ Apply mask correctly
        masked_outputs = outputs[mask]
        masked_labels = labels.expand(-1, -1, 33)[mask]  # Expand labels before masking

        # ✅ Compute loss only on unmasked values
        loss = criterion(masked_outputs, masked_labels)


        # ✅ Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(dataloader)}")

# ✅ Save the trained model
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")
