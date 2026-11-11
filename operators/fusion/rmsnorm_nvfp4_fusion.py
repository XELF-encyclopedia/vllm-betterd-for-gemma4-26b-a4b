# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project & XELF-encyclopedia

"""
RMSNorm + NVFP4 Block Quantization Fusion Pass.

Background:
Standard vLLM executes RMSNorm followed by NVFP4 block quantization as two separate kernels.
This creates high global memory round-trips:
1. `rms_norm(x)` writes FP16/BF16 activations to HBM.
2. `scaled_fp4_quant(normed_x)` reads FP16/BF16 from HBM, calculates block scales, and writes FP4 + E4M3 scales.

This module provides the graph compilation pattern `RMSNormNvfp4QuantPattern` that fuses both
operations into a single FlashInfer `rmsnorm_fp4quant` CUDA kernel launch, cutting memory bandwidth by ~50%.
"""

from typing import Any, Optional
import torch

from torch._dynamo.backends.registry import register_backend
from torch._inductor.pattern_matcher import PatternMatcherPass
from vllm.compilation.fusion import auto_functionalized
from vllm.platforms import current_platform


def register_rms_norm_nvfp4_fusion_pattern(
    pm_pass: PatternMatcherPass,
    fused_op: Any,
    epsilon: float,
    is_sf_swizzled_layout: bool,
):
    """Registers the fused RMSNorm + NVFP4 pattern into PyTorch Inductor / vLLM pattern matcher."""

    def pattern(
        result: torch.Tensor,
        result_block_scale: torch.Tensor,
        input: torch.Tensor,
        weight: torch.Tensor,
        input_global_scale: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        import vllm.ir.ops
        result_rms = vllm.ir.ops.rms_norm(input, weight, epsilon)
        at = auto_functionalized(
            torch.ops._C.scaled_fp4_quant.out,
            input=result_rms,
            input_scale=input_global_scale,
            is_sf_swizzled_layout=is_sf_swizzled_layout,
            output=result,
            output_scale=result_block_scale,
        )
        return at[1], at[2]

    def replacement(
        result: torch.Tensor,
        result_block_scale: torch.Tensor,
        input: torch.Tensor,
        weight: torch.Tensor,
        input_global_scale: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_size = input.shape[-1]
        num_tokens = input.numel() // hidden_size
        block_scale_unswizzled = torch.empty(
            (num_tokens, hidden_size // 16),
            dtype=torch.float8_e4m3fn,
            device=input.device,
        )
        at = auto_functionalized(
            fused_op,
            result=result,
            result_block_scale=result_block_scale,
            input=input,
            weight=weight,
            input_global_scale=input_global_scale,
            block_scale_unswizzled=block_scale_unswizzled,
            is_sf_swizzled_layout=is_sf_swizzled_layout,
            epsilon=epsilon,
        )
        return at[1], at[2]

    return pattern, replacement
