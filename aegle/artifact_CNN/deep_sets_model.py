import torch
import torch.nn as nn
import torchvision.models as models

from aegle.artifact_CNN.resnet_adapter import inject_adapter


class DeepSetsCNN(nn.Module):
    """
    Deep Sets architecture for multi-channel images.
    - Shared backbone processes each channel independently.
    - Channel-specific embeddings are added to the features.
    - Features are pooled (mean) across channels using scatter_add.
    - Final classification head.
    - Supports PEFT (Bottleneck Adapters).
    """

    def __init__(
        self,
        num_antibodies: int,
        num_classes: int,
        backbone_name: str = "resnet50",
        feature_dim: int = 2048,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        use_peft: bool = False,
    ):
        super().__init__()

        # 1. Shared Backbone
        if backbone_name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet50(weights=weights)
            backbone_out_dim = 2048
            adapter_dim = 64
        elif backbone_name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet18(weights=weights)
            backbone_out_dim = 512
            adapter_dim = 16
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        # 2. PEFT / Freezing Logic
        if use_peft:
            # Inject adapters into residual blocks
            for name, module in self.backbone.named_modules():
                if isinstance(
                    module, (models.resnet.BasicBlock, models.resnet.Bottleneck)
                ):
                    inject_adapter(module, bottleneck_dim=adapter_dim)

            # Freeze backbone but keep adapters trainable
            for name, param in self.backbone.named_parameters():
                if "adapter" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        elif freeze_backbone:
            # Freeze entire backbone
            for param in self.backbone.parameters():
                param.requires_grad = False

        # 3. Modify Input Layer (conv1) to 1 channel
        # This creates a NEW layer, so it will be trainable by default (requires_grad=True)
        old_conv = self.backbone.conv1
        new_conv = nn.Conv2d(
            1,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias,
        )

        # Initialize weights: Average the RGB weights
        with torch.no_grad():
            new_conv.weight[:, 0, :, :] = torch.mean(old_conv.weight, dim=1)

        self.backbone.conv1 = new_conv

        # 4. Remove the original fc layer
        self.backbone.fc = nn.Identity()

        # 5. Antibody Embedding
        self.feature_dim = backbone_out_dim
        self.antibody_embedding = nn.Embedding(num_antibodies, self.feature_dim)

        # 6. Classifier
        self.layer_norm = nn.LayerNorm(self.feature_dim)
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward(
        self, x: torch.Tensor, antibody_ids: torch.Tensor, batch_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            x: (Total_C, 1, H, W) - Flattened channels from all images in batch
            antibody_ids: (Total_C,) - Antibody IDs for each channel
            batch_indices: (Total_C,) - Batch index (0..B-1) for each channel
        """
        # 1. Backbone feature extraction
        # x is already (Total_C, 1, H, W)
        features = self.backbone(x)  # (Total_C, feature_dim)

        # 2. Add Antibody Embeddings
        # antibody_ids: (Total_C,) -> embeddings: (Total_C, feature_dim)
        embeddings = self.antibody_embedding(antibody_ids)
        features = features + embeddings

        # 3. Deep Sets Pooling (Max) using Scatter
        # Need to pool features based on batch_indices.
        # Determine batch size B from batch_indices (max index + 1)
        B = batch_indices.max().item() + 1

        # Initialize pooled_features with a very small number for max pooling
        pooled_features = torch.full(
            (B, self.feature_dim), -1e9, device=x.device, dtype=features.dtype
        )
        counts = torch.zeros(B, device=x.device, dtype=features.dtype)

        # Scatter Max
        # Need to expand batch_indices to match feature_dim
        # batch_indices: (Total_C,) -> (Total_C, feature_dim)
        indices_expanded = batch_indices.unsqueeze(1).expand(-1, self.feature_dim)

        pooled_features.scatter_reduce_(
            0, indices_expanded, features, reduce="amax", include_self=False
        )

        # Apply LayerNorm
        if hasattr(self, "layer_norm"):
            pooled_features = self.layer_norm(pooled_features)

        # 4. Classification
        logits = self.classifier(pooled_features)  # (B, num_classes)

        return logits
