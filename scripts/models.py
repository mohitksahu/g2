"""
Model architectures:
  1. ImageBranch    — EfficientNet-B4 feature extractor
  2. TabularBranch  — MLP for clinical tabular data
  3. CrossAttentionFusion — fuses image + tabular embeddings
  4. DRMultiModalNet      — full multi-task model (grade + progression)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


# ──────────────────────────────────────────────
# 1. CNN Branch (Image)
# ──────────────────────────────────────────────
class ImageBranch(nn.Module):
    """
    Pretrained EfficientNet-B4 as feature extractor.
    Removes classification head, outputs 1792-d embedding.
    """

    def __init__(self, backbone_name: str = "efficientnet_b4",
                 pretrained: bool = True, freeze_ratio: float = 0.7):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0
        )
        # num_classes=0 removes the classifier head, returns pooled features

        # Freeze early layers (transfer learning)
        params = list(self.backbone.parameters())
        freeze_count = int(len(params) * freeze_ratio)
        for param in params[:freeze_count]:
            param.requires_grad = False

        self.embed_dim = self.backbone.num_features  # 1792 for EfficientNet-B4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 224, 224) fundus image tensor
        Returns:
            (B, 1792) image embedding
        """
        return self.backbone(x)

    def get_feature_extractor(self):
        """Return the backbone for Grad-CAM target layer access."""
        return self.backbone


# ──────────────────────────────────────────────
# 2. Tabular Branch (MLP)
# ──────────────────────────────────────────────
class TabularBranch(nn.Module):
    """
    MLP to process structured clinical/EHR features.
    Input dim varies based on available features.
    """

    def __init__(self, input_dim: int, embed_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
        )
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, input_dim) tabular feature tensor
        Returns:
            (B, embed_dim) tabular embedding
        """
        return self.net(x)


# ──────────────────────────────────────────────
# 3. Cross-Attention Fusion
# ──────────────────────────────────────────────
class CrossAttentionFusion(nn.Module):
    """
    Cross-attention mechanism to fuse image and tabular embeddings.
    Image attends to tabular and vice versa, then combined.
    More expressive than simple concatenation.
    """

    def __init__(self, img_dim: int, tab_dim: int, fusion_dim: int = 256,
                 num_heads: int = 4, dropout: float = 0.1):
        super().__init__()

        # Project both modalities to same dimension
        self.img_proj = nn.Linear(img_dim, fusion_dim)
        self.tab_proj = nn.Linear(tab_dim, fusion_dim)

        # Cross-attention: image queries, tabular keys/values
        self.cross_attn_img = nn.MultiheadAttention(
            embed_dim=fusion_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True
        )

        # Cross-attention: tabular queries, image keys/values
        self.cross_attn_tab = nn.MultiheadAttention(
            embed_dim=fusion_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True
        )

        self.layer_norm = nn.LayerNorm(fusion_dim)
        self.fusion_dim = fusion_dim * 2  # concatenation of both attended outputs

    def forward(self, img_embed: torch.Tensor,
                tab_embed: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img_embed: (B, img_dim) image embedding
            tab_embed: (B, tab_dim) tabular embedding
        Returns:
            (B, fusion_dim*2) fused embedding
        """
        # Project
        img_proj = self.img_proj(img_embed).unsqueeze(1)   # (B, 1, fusion_dim)
        tab_proj = self.tab_proj(tab_embed).unsqueeze(1)   # (B, 1, fusion_dim)

        # Cross-attention (image attending to tabular)
        img_attended, _ = self.cross_attn_img(
            query=img_proj, key=tab_proj, value=tab_proj
        )
        img_attended = self.layer_norm(img_attended + img_proj)

        # Cross-attention (tabular attending to image)
        tab_attended, _ = self.cross_attn_tab(
            query=tab_proj, key=img_proj, value=img_proj
        )
        tab_attended = self.layer_norm(tab_attended + tab_proj)

        # Concatenate both attended representations
        fused = torch.cat([
            img_attended.squeeze(1),
            tab_attended.squeeze(1)
        ], dim=-1)  # (B, fusion_dim*2)

        return fused


# ──────────────────────────────────────────────
# 4. Simple Concatenation Fusion (Baseline)
# ──────────────────────────────────────────────
class ConcatFusion(nn.Module):
    """Simple concatenation fusion as a baseline comparison."""

    def __init__(self, img_dim: int, tab_dim: int, fusion_dim: int = 256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(img_dim + tab_dim, fusion_dim),
            nn.BatchNorm1d(fusion_dim),
            nn.ReLU(),
        )
        self.fusion_dim = fusion_dim

    def forward(self, img_embed: torch.Tensor,
                tab_embed: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([img_embed, tab_embed], dim=-1)
        return self.proj(combined)


# ──────────────────────────────────────────────
# 5. Full Multi-Modal Multi-Task Model
# ──────────────────────────────────────────────
class DRMultiModalNet(nn.Module):
    """
    Complete model:
      Image → CNN → embedding
      Tabular → MLP → embedding
      Fusion (cross-attention or concat)
      → DR grade head (5-class classification)
      → Progression risk head (binary/regression)
    """

    def __init__(
        self,
        tabular_input_dim: int,
        cnn_backbone: str = "efficientnet_b4",
        cnn_pretrained: bool = True,
        tabular_embed_dim: int = 128,
        fusion_dim: int = 256,
        num_classes: int = 5,
        dropout: float = 0.3,
        fusion_type: str = "cross_attention",  # "cross_attention" or "concat"
        freeze_ratio: float = 0.7,
    ):
        super().__init__()

        # Branches
        self.image_branch = ImageBranch(
            backbone_name=cnn_backbone,
            pretrained=cnn_pretrained,
            freeze_ratio=freeze_ratio,
        )
        self.tabular_branch = TabularBranch(
            input_dim=tabular_input_dim,
            embed_dim=tabular_embed_dim,
            dropout=dropout,
        )

        # Fusion
        img_dim = self.image_branch.embed_dim
        tab_dim = self.tabular_branch.embed_dim

        if fusion_type == "cross_attention":
            self.fusion = CrossAttentionFusion(
                img_dim=img_dim, tab_dim=tab_dim,
                fusion_dim=fusion_dim, dropout=dropout / 3,
            )
            fused_dim = self.fusion.fusion_dim
        else:
            self.fusion = ConcatFusion(
                img_dim=img_dim, tab_dim=tab_dim,
                fusion_dim=fusion_dim,
            )
            fused_dim = self.fusion.fusion_dim

        # Output heads
        self.grade_head = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

        self.progression_head = nn.Sequential(
            nn.Linear(fused_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        self.fusion_type = fusion_type

    def forward(self, image: torch.Tensor, tabular: torch.Tensor):
        """
        Args:
            image:   (B, 3, 224, 224)
            tabular: (B, tabular_input_dim)
        Returns:
            grade_logits:     (B, 5) — DR grade logits
            progression_risk: (B, 1) — 12-month progression probability
        """
        img_embed = self.image_branch(image)       # (B, 1792)
        tab_embed = self.tabular_branch(tabular)   # (B, 128)

        fused = self.fusion(img_embed, tab_embed)  # (B, fused_dim)

        grade_logits = self.grade_head(fused)       # (B, 5)
        progression_risk = self.progression_head(fused)  # (B, 1)

        return grade_logits, progression_risk

    def get_image_backbone(self):
        """Access CNN backbone for Grad-CAM."""
        return self.image_branch.get_feature_extractor()


# ──────────────────────────────────────────────
# 6. Image-Only Baseline (for ablation study)
# ──────────────────────────────────────────────
class DRImageOnlyNet(nn.Module):
    """Baseline: image-only model without tabular data (for comparison)."""

    def __init__(self, cnn_backbone: str = "efficientnet_b4",
                 pretrained: bool = True, num_classes: int = 5,
                 dropout: float = 0.3):
        super().__init__()
        self.image_branch = ImageBranch(cnn_backbone, pretrained)

        self.grade_head = nn.Sequential(
            nn.Linear(self.image_branch.embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

        self.progression_head = nn.Sequential(
            nn.Linear(self.image_branch.embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, image: torch.Tensor, tabular: torch.Tensor = None):
        """tabular is accepted but ignored (API compatibility)."""
        img_embed = self.image_branch(image)
        grade_logits = self.grade_head(img_embed)
        progression_risk = self.progression_head(img_embed)
        return grade_logits, progression_risk
