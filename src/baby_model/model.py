import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, num_layers, dropout=0.1):
        super().__init__()
        self.input_layer = nn.Linear(33, embed_dim)  # ✅ Project input to embed_dim
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim, dropout=dropout),
            num_layers=num_layers
        )
        self.output_layer = nn.Linear(embed_dim, 33)  # ✅ Project back to original feature size

    def forward(self, x):
        batch_size = x.shape[0]

        # ✅ Reshape to (batch_size, 96, 32)
        x = x.view(batch_size, 96, 33)  

        x = self.input_layer(x)  # ✅ Project features to embed_dim
        x = self.transformer(x)  # ✅ Apply Transformer Encoder
        x = self.output_layer(x)  # ✅ Project back to (batch_size, 96, 32)

        return x  # ✅ Keep structure (batch_size, 96, 32) instead of flattening
