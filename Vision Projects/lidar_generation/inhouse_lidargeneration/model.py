import torch
import torch.nn as nn

class DepthEstimationModel(nn.Module):
    def __init__(self):
        super(DepthEstimationModel, self).__init__()
        # Encoder: Convolutional layers for feature extraction
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        # Attention Mechanism: Self-attention to capture global context
        self.attention = nn.MultiheadAttention(embed_dim=256, num_heads=8)

        # Decoder: Upsampling layers for depth map generation
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        # Encoder
        features = self.encoder(x)

        # Attention
        b, c, h, w = features.size()
        features_flat = features.view(b, c, -1).permute(2, 0, 1)  # Flatten for attention
        attended_features, _ = self.attention(features_flat, features_flat, features_flat)
        attended_features = attended_features.permute(1, 2, 0).view(b, c, h, w)

        # Decoder
        depth = self.decoder(attended_features)
        return depth
