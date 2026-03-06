"""Contains Dense Transformer Prediction architecture.

Implements a variant of Vision Transformers for Dense Prediction, https://arxiv.org/abs/2103.13413

For licensing see accompanying LICENSE file.
Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn

from sharp.models.blocks import (
    FeatureFusionBlock2d,
    NormLayerName,
    residual_block_2d,
)
from sharp.models.decoders import BaseDecoder, MultiresConvDecoder
from sharp.models.params import DPTImageEncoderType, GaussianDecoderParams
from sharp.models.gaussian_decoder import create_gaussian_decoder

from sharp.models.presets import (
    MONODEPTH_ENCODER_DIMS_MAP,
    MONODEPTH_HOOK_IDS_MAP,
    ViTPreset,
)
from IPython import embed

class GaussianDensePredictionTransformerWrap(nn.Module):
    def __init__(self, feature_decoder):
        super().__init__()
        self.feature = feature_decoder
    
    def forward(self, input, feature_0, feature_1, feature_2, feature_3, feature4):
        out = self.feature(input, [feature_0, feature_1, feature_2, feature_3, feature4])
        return out

if __name__ == '__main__':
    patch_encoder_preset = 'dinov2s16_384'

    params = GaussianDecoderParams()
    params.patch_encoder_preset = patch_encoder_preset

    dims_encoder = [128] + MONODEPTH_ENCODER_DIMS_MAP[patch_encoder_preset]

    feature_model = create_gaussian_decoder(params, dims_encoder)

    feature_wrap = GaussianDensePredictionTransformerWrap(feature_model)

    # rgn + 2d
    example_input = [torch.rand(1,5,1536, 1536), torch.rand(1, 128, 768, 768), torch.rand(1, 96, 384, 384), torch.rand(1, 192, 192, 192), 
        torch.rand(1, 384, 96, 96), torch.rand(1, 384, 48, 48)]

    feature = feature_wrap(*example_input) # (2, 32, 768, 768) texture, geometry

    import coremltools as ct
    with torch.no_grad():
        model = feature_wrap.eval()
        example_input = [torch.rand(1,5,1536, 1536), torch.rand(1, 128, 768, 768), torch.rand(1, 96, 384, 384), torch.rand(1, 192, 192, 192), 
        torch.rand(1, 384, 96, 96), torch.rand(1, 384, 48, 48)]
        traced_model = torch.jit.trace(model, example_input)

        inputs = [ct.TensorType(shape=x.shape) for x in example_input]
        
        model_from_trace = ct.convert(
            traced_model,
            inputs=inputs,
            convert_to="mlprogram",
            compute_precision=ct.precision.FLOAT16,
        )
        model_from_trace.save("models/feature_vits.mlpackage")

        loaded_model = ct.models.MLModel("models/feature_vits.mlpackage", optimization_hints ={"specializationStrategy":ct.SpecializationStrategy.FastPrediction}) 


    embed()

# python -m sharp.models.gaussian_decoder_wrap