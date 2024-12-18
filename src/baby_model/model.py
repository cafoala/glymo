import torch
import torch.nn as nn

class MaskedCGMTransformer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, num_layers, dropout=0.1):
        super().__init__()
        self.input_layer = nn.Linear(288, embed_dim)  # Input layer
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim, dropout=dropout),
            num_layers=num_layers
        )
        self.output_layer = nn.Linear(embed_dim, 288)  # Output layer

    def forward(self, x):
        x = self.input_layer(x)  # Project input
        x = self.transformer(x)  # Apply Transformer
        x = self.output_layer(x)  # Project back to original dimension
        return x
