import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import VideoMAEModel


class EchoNetRegressor(nn.Module):
    """
    EchoNet EF Prediction Model with:
    - VideoMAE backbone
    - Attention Pooling
    - Deeper MLP head (768 -> 256 -> 64 -> 1)
    - Confidence estimation via MC Dropout
    """
    def __init__(self, pretrained_model="MCG-NJU/videomae-base", freeze_backbone=False):
        super().__init__()
        
        print(f"Loading VideoMAE Backbone: {pretrained_model}")
        self.backbone = VideoMAEModel.from_pretrained(pretrained_model)
        
        if freeze_backbone:
            print("Freezing Backbone Weights...")
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        hidden_size = self.backbone.config.hidden_size  # 768 for base
        
        # Attention Pooling
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        # Deeper MLP Head with Dropout for MC Dropout confidence estimation
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )
        
        # Initialize final layer bias to mean EF (~55%)
        self.head[-1].bias.data.fill_(55.0)
        
        # Store attention weights for Grad-CAM visualization
        self._attn_weights = None

    def forward(self, pixel_values, return_attention: bool = False):
        """
        Forward pass with optional attention weight return for visualization.
        
        Args:
            pixel_values: (B, F, C, H, W) tensor
            return_attention: If True, return attention weights for Grad-CAM
        """
        outputs = self.backbone(pixel_values)
        last_hidden_state = outputs.last_hidden_state  # (B, SeqLen, 768)
        
        # Attention Pooling
        attn_scores = self.attention(last_hidden_state)  # (B, SeqLen, 1)
        attn_weights = F.softmax(attn_scores, dim=1)     # (B, SeqLen, 1)
        self._attn_weights = attn_weights  # Store for visualization
        
        feat = (last_hidden_state * attn_weights).sum(dim=1)  # (B, 768)
        ef_prediction = self.head(feat)
        
        if return_attention:
            return ef_prediction, attn_weights
        return ef_prediction

    def predict_with_confidence(self, pixel_values, n_samples: int = 10) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Monte Carlo Dropout for confidence estimation.
        Run multiple forward passes with dropout enabled.
        
        Args:
            pixel_values: Input tensor
            n_samples: Number of MC samples
            
        Returns:
            mean_pred: Mean prediction across samples
            confidence: 1 / std (higher = more confident)
        """
        self.train()  # Enable dropout
        
        predictions = []
        for _ in range(n_samples):
            with torch.no_grad():
                pred = self.forward(pixel_values)
                predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=0)  # (n_samples, B, 1)
        
        mean_pred = predictions.mean(dim=0)
        std_pred = predictions.std(dim=0)
        
        # Confidence: inverse of std (clamped to avoid div by zero)
        confidence = 1.0 / (std_pred + 0.1)
        
        self.eval()  # Restore eval mode
        return mean_pred, confidence

    def get_attention_weights(self) -> torch.Tensor | None:
        """Return last computed attention weights for visualization."""
        return self._attn_weights


if __name__ == "__main__":
    model = EchoNetRegressor(freeze_backbone=True)
    print("Model initialized.")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    
    dummy_input = torch.randn(2, 16, 3, 224, 224)
    output = model(dummy_input)
    print(f"Output Shape: {output.shape}")
    
    # Test confidence estimation
    mean, conf = model.predict_with_confidence(dummy_input, n_samples=5)
    print(f"MC Dropout - Mean: {mean.flatten()}, Confidence: {conf.flatten()}")
