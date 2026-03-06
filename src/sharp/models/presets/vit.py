"""Contains preset for ViT modules.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

import dataclasses
from typing import Literal

ViTPreset = Literal["dinov2l16_384",]

MLPMode = Literal["vanilla", "glu"]


@dataclasses.dataclass
class ViTConfig:
    """Configuration for ViT."""

    in_chans: int
    embed_dim: int
    depth: int
    num_heads: int
    init_values: float

    img_size: int = 384
    patch_size: int = 16

    num_classes: int = 21841
    mlp_ratio: float = 4.0
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0
    drop_path_rate: float = 0.0
    qkv_bias: bool = True
    global_pool: str = "avg"

    # Properties for timm_vit.
    mlp_mode: MLPMode = "vanilla"

    # Properties for SPN.
    intermediate_features_ids: list[int] | None = None

    def asdict(self):
        """Convenience method to convert the class to a dict."""
        return dataclasses.asdict(self)


VIT_CONFIG_DICT: dict[ViTPreset, ViTConfig] = {
    "dinov2l16_384": ViTConfig(
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        init_values=1e-5,
        global_pool="",
    ),
        # ViT-Base(86M) - dinov2b16_384
    "dinov2b16_384": ViTConfig(
        in_chans=3,
        embed_dim=768,  
        depth=12,       
        num_heads=12,  
        init_values=1e-5,
        global_pool="",
    ),
    
    # ViT-Samll - dinov2s16_384
    "dinov2s16_384": ViTConfig(
        in_chans=3,
        embed_dim=384,
        depth=12,    
        num_heads=6,    
        init_values=1e-5,
        global_pool="",
    ),
}
