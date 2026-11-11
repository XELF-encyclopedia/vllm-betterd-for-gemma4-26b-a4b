# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project & XELF-encyclopedia

"""
Heterogeneous Hybrid Attention Engine for Gemma4 on Blackwell SM120.

Background:
Gemma4-26B-A4B features heterogeneous attention heads across layers:
- Sliding Window Attention (SWA): head_dim = 256
- Full Global Attention: head_dim = 512

FlashInfer XQA decode kernels natively support head_dim <= 256. For head_dim = 512,
standard FlashInfer reverts the entire KV-cache group to native slow fallback paths.
Furthermore, allocating CPU metadata buffers per step introduces noticeable CPU-GPU sync latency.

This module implements:
1. Hybrid Dispatch: Maintains the unified XQA metadata pipeline across all layers.
   - Layers with head_dim = 256 execute FlashInfer XQA TensorCore decode.
   - Layers with head_dim = 512 execute Triton Unified Attention 3D decode using pre-built XQA metadata.
2. Pinned CPU Host Staging Buffers: Pre-allocates pinned memory buffers (`_pinned_qo_indptr_prefill`,
   `_pinned_kv_indptr_prefill`, etc.) to eliminate per-step CPU allocation and achieve zero-copy DMA.
3. Pre-allocated 3D Decode Workspaces: Minimizes dynamic allocation in Triton multi-segment softmax.
"""

from typing import Any, Optional
import torch

from vllm.platforms import current_platform
from vllm.v1.attention.backends.flashinfer import FlashInferImpl
from vllm.v1.attention.ops.triton_unified_attention import unified_attention


class HeterogeneousHybridAttentionImpl:
    """Helper dispatcher demonstrating the hybrid XQA + Triton Unified Attention decode flow."""

    @staticmethod
    def dispatch_decode(
        impl: FlashInferImpl,
        query: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        output: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        attn_metadata: Any,
        layer: Any,
    ) -> torch.Tensor:
        num_tokens = query.shape[0]
        head_size = impl.head_size

        # If head_size == 512 or large batch decode without NVFP4 KV cache:
        if not impl._xqa_head_dim_supported or (
            not impl.is_kvcache_nvfp4
            and attn_metadata.decode.q_cu_seq_lens is None
            and num_tokens >= 24
        ):
            num_seqs = seq_lens.shape[0]
            descale_shape = (num_seqs, impl.num_kv_heads)
            k_descale = getattr(layer, "_k_scale", None)
            v_descale = getattr(layer, "_v_scale", None)
            if k_descale is not None:
                k_descale = k_descale.expand(descale_shape)
                v_descale = v_descale.expand(descale_shape)

            cu_seqlens_q = (
                attn_metadata.decode.q_cu_seq_lens
                if attn_metadata.decode.q_cu_seq_lens is not None
                else impl._TRITON_CU_SEQLENS_Q[query.device][: num_seqs + 1]
            )

            # 3D softmax parallel segment optimization for SM120
            if not impl._xqa_head_dim_supported and num_seqs <= 64:
                seq_threshold_3D = 64
                s_out_base, s_max_base, s_exp_base = impl._TRITON_3D_BUFFERS[
                    (query.device, impl.num_heads, impl.head_size)
                ]
                if num_seqs <= 4:
                    segs = 64
                    s_out, s_max, s_exp = s_out_base, s_max_base, s_exp_base
                else:
                    segs = 16
                    s_out = s_out_base.view(64, impl.num_heads, 16, impl.head_size)
                    s_max = s_max_base.view(64, impl.num_heads, 16)
                    s_exp = s_exp_base.view(64, impl.num_heads, 16)
            else:
                seq_threshold_3D = None
                segs = None
                s_out = s_max = s_exp = None

            unified_attention(
                q=query,
                k=k_cache,
                v=v_cache,
                out=output[:num_tokens],
                cu_seqlens_q=cu_seqlens_q,
                max_seqlen_q=1,
                seqused_k=seq_lens,
                max_seqlen_k=attn_metadata.decode.max_seq_len,
                softmax_scale=impl.scale,
                causal=True,
                window_size=impl.sliding_window,
                block_table=block_tables,
                softcap=impl.logits_soft_cap or 0,
                q_descale=None,
                k_descale=k_descale,
                v_descale=v_descale,
                seq_threshold_3D=seq_threshold_3D,
                num_par_softmax_segments=segs,
                softmax_segm_output=s_out,
                softmax_segm_max=s_max,
                softmax_segm_expsum=s_exp,
                sinks=impl.sinks,
                kv_quant_mode=impl._kv_quant_mode,
            )
            return output
        return output
