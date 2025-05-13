import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, embed_dim=64, num_heads=8, ff_dim=256, num_layers=4, dropout=0.2):
        super().__init__()
        
        # ✅ Input projection layer
        self.input_layer = nn.Linear(33, embed_dim)  

        # ✅ Transformer layers with normalization
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='gelu',  # ✅ GELU is better than ReLU for transformers
            batch_first=True  # ✅ Ensures input format (batch_size, seq_len, embed_dim)
        )

        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        # ✅ Layer Normalization
        self.norm = nn.LayerNorm(embed_dim)
        
        # ✅ Output projection layer
        self.output_layer = nn.Linear(embed_dim, 33)  

    def forward(self, x):
        batch_size = x.shape[0]

        # ✅ Ensure correct shape (batch_size, 96, 33)
        x = x.view(batch_size, 96, 33)  

        # ✅ Input projection
        x = self.input_layer(x)  

        # ✅ Normalize before transformer
        x = self.norm(x)

        # ✅ Apply Transformer Encoder with residual connection
        x = x + self.transformer(x)

        # ✅ Final projection back to original dimension
        x = self.output_layer(x)

        return x
