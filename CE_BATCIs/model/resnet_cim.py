import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# from einops import rearrange


class ContralateralInteractionModule(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.eca = ECAModule(channels)
        self.kv_proj = nn.Conv3d(2*channels, channels, kernel_size=1)

    def forward(self, left_feat, right_feat):

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
        B = left_patch.size(0)  # （B, 16, 2, 32, 32, 32)
        input_left_patch = left_patch.view(-1, 2, 32, 32, 32)
        input_right_patch = right_patch.view(-1, 2, 32, 32, 32)

        left_feat, right_feat = self.shared_backbone(input_left_patch, input_right_patch) # （B*16, 256, 4, 4, 4)
        combined_features = torch.cat([left_feat, right_feat], dim=0) # （B*32, 256, 4, 4, 4)

        combined_features = self.global_pool(combined_features) # B*32, 256, 1, 1, 1)
        out_feats =combined_features.view(B, 32, -1)
        preds = self.classifier_head(combined_features.view(B*32, -1))
        return preds, out_feats


class CaseLevelBATCls(nn.Module):

    def __init__(self, num_patches=32, embed_dim=256):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = PositionalEncoding3D(embed_dim)
        self.patch_proj = nn.Linear(256, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=8, dim_feedforward=1024, dropout=0.1)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, patch_features):
        # patch_features: [B, 32, 256]
        batch_size = patch_features.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)

        x = self.patch_proj(patch_features)  # [B, 32, 256] -> [B, 32, embed_dim]
        x = self.pos_embed(x)
        x = torch.cat([cls_tokens, x], dim=1)  # [B, 33, embed_dim]

        x = self.transformer(x)

        cls_output = x[:, 0, :]
        return self.classifier(cls_output)


class PositionalEncoding3D(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))

    def forward(self, x):
        # x: [B, 32, C]
        B, N, C = x.shape
        assert N == 32

        pos = torch.zeros(1, 2, 2, 8, self.dim, device=x.device)

        for z in range(2):
            for y in range(2):
                for x in range(4):
                    pos[0, z, y, x, 0::2] = torch.sin((z * 16 + y * 8 + x) * self.div_term)
                    pos[0, z, y, x, 1::2] = torch.cos((z * 16 + y * 8 + x) * self.div_term)
        for z in range(2):
            for y in range(2):
                for x in range(7, 3, -1):
                    pos[0, z, y, x, 0::2] = torch.sin((z * 16 + y * 8 + x) * self.div_term)
                    pos[0, z, y, x, 1::2] = torch.cos((z * 16 + y * 8 + x) * self.div_term)

        pos = pos.view(1, -1, self.dim).expand(B, -1, -1)
        return x + pos


class CEBATCls(nn.Module):

    def __init__(self):
        super().__init__()
        self.patch_model = PatchLevelBATCls()
        self.case_model = CaseLevelBATCls()

    def forward(self, left_patches, right_patches):
        # left/right_patches: [B, 16, 2, 32, 32, 32]
        patch_preds, patch_features = self.patch_model(left_patches, right_patches)
        case_pred = self.case_model(patch_features)
        return case_pred, patch_preds

