import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ContralateralInteractionModule(nn.Module):
    """
    Cross-lateral interaction module designed to enhance bilateral feature
    communication between left and right 3D feature maps.

    Mechanism:
        - Horizontally flip the right feature map to align anatomical sides.
        - Concatenate the left and flipped-right features.
        - Generate a key-value projection (via 1x1x1 conv).
        - Compute attention maps for both sides to extract symmetrical relations.
        - Enhance both left and right features using attention-guided fusion.
        - Refine outputs with ECA (Efficient Channel Attention).
    """
    def __init__(self, channels):
        super().__init__()
        self.eca = ECAModule(channels)
        self.kv_proj = nn.Conv3d(2*channels, channels, kernel_size=1)

    def forward(self, left_feat, right_feat):
        """
        Args:
            left_feat:  [B, C, D, H, W] feature map from left input
            right_feat: [B, C, D, H, W] feature map from right input
        Returns:
            enhanced_left, enhanced_right: contralaterally refined feature maps
        """
        # Flip right feature map along the width (left-right axis)
        right_flipped = torch.flip(right_feat, [-1])
        kv_feat = torch.cat([left_feat, right_flipped], dim=1)
        kv_feat = self.kv_proj(kv_feat)
        b, c, d, w, h = left_feat.size()

        q_left = left_feat.view(b, c, -1)
        q_right = right_flipped.view(b, c, -1)
        kv_feat = kv_feat.view(b, c, -1)

        attn_left = F.softmax((q_left @ kv_feat.transpose(-2, -1)) / torch.sqrt(torch.tensor(kv_feat.size(-1),
                                                                                             dtype=torch.float)), dim=-1)
        attn_right = F.softmax((q_right @ kv_feat.transpose(-2, -1)) / torch.sqrt(torch.tensor(kv_feat.size(-1),
                                                                                               dtype=torch.float)), dim=-1)

        enhanced_left = (attn_left @ kv_feat).view(b, c, d, w, h) + left_feat
        enhanced_right = (attn_right @ kv_feat).view(b, c, d, w, h) + right_flipped

        enhanced_left = self.eca(enhanced_left)
        enhanced_right = self.eca(enhanced_right)

        enhanced_right = torch.flip(enhanced_right, [-1])

        return enhanced_left, enhanced_right


class ECAModule(nn.Module):
    """
    Efficient Channel Attention module:
    - Avoids dimensionality reduction.
    - Captures local cross-channel interaction using 1D convolution.
    - Lightweight and effective for 3D feature recalibration.
    """
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        k_size = int(abs((math.log2(channels) + b) / gamma))
        k_size = k_size if k_size % 2 else k_size + 1
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1).unsqueeze(-1)
        return x * torch.sigmoid(y)


class ResNet3DBlock(nn.Module):
    """
    Standard 3D ResNet block with two 3x3x3 convolutions and a skip connection.
    Supports stride for downsampling.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm3d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class SharedResNet3D(nn.Module):
    """
    Shared 3D ResNet backbone used for both left and right lung patches.
    Integrates Contralateral Interaction Modules (CIMs) to enhance
    bilateral contextual learning at intermediate feature levels.
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(2, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm3d(64)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.stage1 = ResNet3DBlock(64, 64)
        self.stage2 = ResNet3DBlock(64, 128, stride=2)
        self.stage3 = ResNet3DBlock(128, 256, stride=2)

        self.cim1 = ContralateralInteractionModule(128)
        self.cim2 = ContralateralInteractionModule(256)

    def forward(self, x_left, x_right):
        """
        Args:
            x_left, x_right: [B, 2, D, H, W] input volumes for left/right at
        Returns:
            left, right: deep feature maps after shared ResNet + bilateral enhancement
        """
        left = F.relu(self.bn1(self.conv1(x_left)))
        right = F.relu(self.bn1(self.conv1(x_right)))
        left = self.maxpool(left)
        right = self.maxpool(right)

        left = self.stage1(left)
        right = self.stage1(right)

        left = self.stage2(left)
        right = self.stage2(right)
        left, right = self.cim1(left, right)

        left = self.stage3(left)
        right = self.stage3(right)
        left, right = self.cim2(left, right)

        return left, right


class PatchLevelBATCls(nn.Module):
    """
    Patch-level bilateral attention transformer classifier.
    Extracts 3D local patch features from left/right inputs,
    enhances them through SharedResNet3D + CIMs,
    and produces patch-wise classification logits.
    """
    def __init__(self):
        super().__init__()
        self.shared_backbone = SharedResNet3D()

        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.classifier_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )


    def forward(self, left_patch, right_patch):
        """
        Args:
            left_patch, right_patch: [B, 16, 2, 32, 32, 32]
        Returns:
            patch_preds: [B*32, 2]  patch-level classification logits
            patch_features: [B, 32, 256]  pooled 3D feature embeddings
        """
        B = left_patch.size(0)          # [B, 16, 2, 32, 32, 32]
        P = left_patch.size(1)

        left_patch = left_patch.view(B * P, 2, 32, 32, 32)
        right_patch = right_patch.view(B * P, 2, 32, 32, 32)

        left_feat, right_feat = self.shared_backbone(left_patch, right_patch)  # [B*P, 256, d, h, w]

        # ensure contiguous (defensive)
        if not left_feat.is_contiguous():
            left_feat = left_feat.contiguous()
        if not right_feat.is_contiguous():
            right_feat = right_feat.contiguous()

        # spatial dims
        _, C, d, h, w = left_feat.shape  # C should be 256

        # restore case & patch dims
        left_feat = left_feat.view(B, P, C, d, h, w)
        right_feat = right_feat.view(B, P, C, d, h, w)
        combined = torch.cat([left_feat, right_feat], dim=1)  # [B, 32, 256, d, h, w]

        pooled = self.global_pool(combined)   # [B, 32, 256, 1, 1, 1]
        patch_features = pooled.view(B, 32, 256)

        patch_preds = self.classifier_head(patch_features.view(B * 32, -1))  # [B*32, 2]

        return patch_preds, patch_features


class CaseLevelBATCls(nn.Module):
    """
    Case-level classifier that aggregates all patch embeddings
    using a lightweight Transformer encoder.

    Uses a [CLS] token to derive the holistic representation
    for patient-level classification.
    """
    def __init__(self, num_patches=32, embed_dim=256):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.patch_proj = nn.Linear(embed_dim, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=8, dim_feedforward=1024,
            dropout=0.1, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, 2)
        )

    def forward(self, patch_features):
        """
        Args:
            patch_features: [B, 32, 256]
        Returns:
            case-level prediction: [B, 2]
        """
        # patch_features: [B, 32, 256]
        B = patch_features.shape[0]
        x = self.patch_proj(patch_features)  # Feature embedding tighten
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls, x), dim=1)  # [B, 33, 256]
        x = x + self.pos_embed[:, :x.size(1), :]
        x = self.transformer(x)  # [B, 33, 256]
        cls_output = x[:, 0]  # case-level embedding
        return self.classifier(cls_output)


class CEBATCls(nn.Module):
    """
    Cascade Enhanced Bilateral Attention Transformer Classifier (CE-BATCls).

    Architecture:
        1. Patch-level bilateral attention model extracts localized features.
        2. Case-level transformer aggregates patch embeddings for global decision.

    Returns both case-level and patch-level predictions.
    """
    def __init__(self):
        super().__init__()
        self.patch_model = PatchLevelBATCls()
        self.case_model = CaseLevelBATCls()

    def forward(self, left_patches, right_patches):
        # left/right_patches: [B, 16, 2, 32, 32, 32]
        patch_preds, patch_features = self.patch_model(left_patches, right_patches)
        case_pred = self.case_model(patch_features)
        return case_pred, patch_preds
