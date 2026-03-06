"""Contains multi-res convolutional decoder.

Implements the decoder for Vision Transformers for Dense Prediction, https://arxiv.org/abs/2103.13413

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn

from sharp.models.blocks import FeatureFusionBlock2d, UpsamplingMode
from sharp.utils.training import checkpoint_wrapper

from .base_decoder import BaseDecoder
from .multires_conv_decoder import MultiresConvDecoder

class MultireeConvDecoderWrap(BaseDecoder):
    def __init__(self, decoder: MultiresConvDecoder):
        super().__init__()

        self.decoder = decoder
    
    def forward(self, encoder_0, encoder_1, encoder_2, encoder_3, encoder_4):
        features = self.decoder([encoder_0, encoder_1,encoder_2,encoder_3,encoder_4])

        return features