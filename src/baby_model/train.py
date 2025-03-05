import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from model import TransformerModel  # Ensure this matches (batch_size, 96, 32)
from dataset import MaskedCGMDataset  # Ensure correct reshaping
import numpy as np

# ✅ Hyperparameters
embed_dim = 32
num_heads = 4
num_layers = 2
dropout = 0.1
learning_rate = 1e-4
batch_size = 64  # ✅ Increased for efficiency
epochs = 10
mask_token = -1
ff_dim = 128  # Transformer feedforward dimension
accumulation_steps = 2  # ✅ For gradient accumulation (handles large batch sizes)

# ✅ File paths
masked_file = "data/processed/masked_windows_aleppo.npy"
labels_file = "data/processed/mask_labels_aleppo.npy"

#masked_file = "data/processed/masked_windows_lynch.npy"
#labels_file = "data/processed/mask_labels_lynch.npy"

model_save_path = "models/baby_transformer_cgm.pth"

# ✅ Load dataset
data = np.load(masked_file)
print("Final dataset shape:", data.shape)  # Should be (num_samples, 96, 32)

# ✅ Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Load dataset into DataLoader
dataset = MaskedCGMDataset(masked_file, labels_file, mask_token=mask_token)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# ✅ Initialize Transformer model
model = TransformerModel(embed_dim=embed_dim, num_heads=num_heads, ff_dim=ff_dim, num_layers=num_layers, dropout=dropout)
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs!")
    model = nn.DataParallel(model)  # ✅ Multi-GPU support
model = model.to(device)

# ✅ Verify first batch shape before training
for batch in dataloader:
    inputs, labels = batch
    print("Input shape before passing to model:", inputs.shape)  # Should be (batch_size, 96, 32)
    inputs, labels = inputs.to(device), labels.to(device)
    print("Mask Labels Shape:", labels.shape)
    print("Unique mask label values:", torch.unique(labels))

    # Forward pass
    outputs = model(inputs)
    print("Output shape:", outputs.shape)  # Should match (batch_size, 96, 32)
    print("Sample Outputs:", outputs[0, :5])  # Print first few predictions
    print("Sample Labels:", labels[0, :5].cpu().numpy())
    break  # Exit after one batch for debugging

# ✅ Loss function & optimizer
criterion = nn.SmoothL1Loss()  # ✅ Huber Loss is more robust for glucose
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)  # ✅ AdamW helps with weight decay

# ✅ Track loss per epoch
losses = []

# ✅ Training loop with corrected masking
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    optimizer.zero_grad()

    for step, batch in enumerate(dataloader):
        inputs, labels = batch
        inputs, labels = inputs.to(device), labels.to(device)

        # Forward pass
        outputs = model(inputs)

        # ✅ Ensure mask shape matches (batch_size, 96, 33)
        mask = labels != mask_token  # Create mask for valid values
        mask = mask.expand(-1, -1, outputs.shape[-1])  # ✅ Expand mask to match last dimension

        # ✅ Apply mask correctly
        masked_outputs = outputs[mask]
        masked_labels = labels.expand(-1, -1, outputs.shape[-1])[mask]  # ✅ Expand labels before masking

        # ✅ Compute loss only on unmasked values
        loss = criterion(masked_outputs, masked_labels)
        loss = loss / accumulation_steps  # ✅ Scale loss for accumulation

        # ✅ Backward pass with gradient accumulation
        loss.backward()
        if (step + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()

        epoch_loss += loss.item()

    # ✅ Store loss per epoch
    avg_loss = epoch_loss / len(dataloader)
    losses.append(avg_loss)
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.5f}")


# ✅ Plot loss curve after training
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.show()

# ✅ Save the trained model
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")
