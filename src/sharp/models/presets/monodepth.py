"""Contains preset for monodepth modules.

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

from .vit import ViTPreset

# Map the decoder configuration with the number of output channels
# for each tensor from the decoder output.
MONODEPTH_ENCODER_DIMS_MAP: dict[ViTPreset, list[int]] = {
    # For publication
    "dinov2l16_384": [256, 512, 1024, 1024],
        # ADD
    "dinov2b16_384": [192, 384, 768, 768], # ViT-Base
    "dinov2s16_384": [96, 192, 384, 384], # ViT-Small
    
}

MONODEPTH_HOOK_IDS_MAP: dict[ViTPreset, list[int]] = {
    # For publication
    "dinov2l16_384": [5, 11, 17, 23],

    "dinov2b16_384": [2, 5, 8, 11], # ViT-Base
    "dinov2s16_384": [2, 5, 8, 11], # ViT-Small

}
