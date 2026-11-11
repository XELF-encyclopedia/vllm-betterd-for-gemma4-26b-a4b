# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project & XELF-encyclopedia

"""
FlashInfer CuteDSL + CUTLASS Hybrid Fused MoE Expert Operator
Specially optimized for Blackwell SM120 / SM121 (RTX Pro 6000 / DGX Spark) NVFP4 W4A4.

Key Design:
- Tokens M <= 64 (Decode Phase): Dispatches to FlashInfer CuteDSL B12x micro/static kernel
  with static NVFP4 scaling, delivering up to 1.40x lower latency vs CUTLASS.
- Tokens M > 64 (Prefill / Large Batches): Dynamically switches to CUTLASS grouped GEMM
  for ~20% higher prefill compute throughput.
- Zero-copy In-Place Output: Patched wrapper directly writes to the allocated output buffer.
- Shared Static/Dynamic Workspace: Prevents multi-layer VRAM waste and CUDA graph replay overhead.
"""

from typing import Any
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm._custom_ops import scaled_fp4_quant
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
    activation_to_flashinfer_type,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kNvfp4Dynamic,
    kNvfp4Static,
)
from vllm.platforms import current_platform
from vllm.utils.flashinfer import (
    flashinfer_convert_sf_to_mma_layout,
    flashinfer_cutlass_fused_moe,
    has_flashinfer_b12x_moe,
    has_flashinfer_cutlass_fused_moe,
)

# Crossover batch size on SM120: B12x CuteDSL micro/static kernels are up to
# 1.40x faster for M <= 64 (decode), whereas FlashInfer CUTLASS is ~20% faster
# for M > 64 (prefill / large mixed batches).
B12X_MAX_DECODE_TOKENS = 64


class FlashInferB12xExperts(mk.FusedMoEExpertsModular):
    """FlashInfer CuteDSL fused MoE expert for SM12x (SM120/SM121,
    RTX Pro 6000 / DGX Spark) with hybrid CUTLASS prefill dispatch.
    """

    _ACTIVATION_MAP: dict[MoEActivation, str] = {
        MoEActivation.SILU: "silu",
        MoEActivation.GELU_TANH: "gelu_tanh",
        MoEActivation.RELU2_NO_MUL: "relu2",
    }

    # Shared workspace cache keyed by (device, E, topk, H, I, max_tokens, act)
    _SHARED_WORKSPACES: dict[tuple[Any, ...], tuple[Any, Any, torch.Tensor]] = {}

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ):
        super().__init__(moe_config=moe_config, quant_config=quant_config)
        assert quant_config.quant_dtype == "nvfp4", (
            "FlashInferB12xExperts only supports nvfp4 quantization."
        )
        self.out_dtype = moe_config.in_dtype
        self.num_local_experts = moe_config.num_local_experts
        self.ep_rank = moe_config.moe_parallel_config.ep_rank
        self._fc1_input_scale: torch.Tensor | None = None
        self._fc2_input_scale: torch.Tensor | None = None

        # Shape params for B12xMoEWrapper construction.
        self.global_num_experts = moe_config.num_experts
        self.topk = moe_config.experts_per_token
        self.hidden_dim = moe_config.hidden_dim
        self.intermediate_size_per_partition = (
            moe_config.intermediate_size_per_partition
        )
        self.max_num_tokens = moe_config.max_num_tokens
        self.local_expert_offset = self.ep_rank * self.num_local_experts

        activation = moe_config.activation
        if activation not in self._ACTIVATION_MAP:
            raise ValueError(
                f"FlashInferB12xExperts does not support "
                f"activation {activation!r}. "
                f"Supported: {list(self._ACTIVATION_MAP.keys())}"
            )
        self._activation_str = self._ACTIVATION_MAP[activation]
        self._cutlass_activation_type = activation_to_flashinfer_type(activation)

        def _per_expert(value: float | None) -> torch.Tensor | None:
            if value is None:
                return None
            return torch.full(
                (self.num_local_experts,),
                float(value),
                dtype=torch.float32,
                device=moe_config.device,
            )

        self.gemm1_clamp_limit = _per_expert(quant_config.gemm1_clamp_limit)
        self.gemm1_alpha = _per_expert(quant_config.gemm1_alpha)
        self.gemm1_beta = _per_expert(quant_config.gemm1_beta)

        self._wrapper: Any | None = None
        self.w1_sf_mma: torch.Tensor | None = None
        self.w2_sf_mma: torch.Tensor | None = None
        self._cutlass_quant_scales: list[torch.Tensor] | None = None
        self._can_use_cutlass_prefill: bool = False

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Keep w13_weight_scale in its full-dynamic-range FP8 E4M3 representation and
        # pass w13_weight_scale_2 (g1_alphas) as the exact FP32 w1_alpha multiplier,
        # while passing 1.0 / a1_gscale as input_global_scale for FC1 quantization.
        if self.a1_gscale is not None:
            self._fc1_input_scale = (
                (1.0 / self.a1_gscale).to(torch.float32).contiguous()
            )
        else:
            self._fc1_input_scale = None

        # Do NOT overwrite self.a2_gscale in-place so that the CUTLASS prefill
        # path retains the calibrated static a2_gscale, while B12x uses its
        # dynamic per-block FC2 input scale (uniform 1.0).
        self._fc2_input_scale = torch.ones(
            self.num_local_experts,
            device=layer.w13_weight.device,
            dtype=torch.float32,
        )

        # Precompute CUTLASS-compatible quant_scales for M > B12X_MAX_DECODE_TOKENS.
        if (
            self.quant_config.use_nvfp4_w4a4
            and has_flashinfer_cutlass_fused_moe()
            and self.a1_gscale is not None
            and self.a2_gscale is not None
            and hasattr(layer, "w13_input_scale")
            and hasattr(layer, "w2_input_scale")
        ):
            assert self.g1_alphas is not None and self.g2_alphas is not None
            assert self.w1_scale is not None and self.w2_scale is not None
            cutlass_g1_alphas = (
                (self.g1_alphas * layer.w13_input_scale).to(torch.float32).contiguous()
            )
            cutlass_g2_alphas = (
                (self.g2_alphas * layer.w2_input_scale).to(torch.float32).contiguous()
            )
            self._cutlass_quant_scales = [
                self.a1_gscale,
                self.w1_scale.view(torch.int32),
                cutlass_g1_alphas,
                self.a2_gscale,
                self.w2_scale.view(torch.int32),
                cutlass_g2_alphas,
            ]
            self._can_use_cutlass_prefill = True

        # Precompute MMA-layout views of the weight scale factors once here
        # rather than recomputing on every forward pass.
        assert self.w1_scale is not None
        num_experts_w1, m1, k1_sf = self.w1_scale.shape
        k1 = k1_sf * 16
        self.w1_sf_mma = flashinfer_convert_sf_to_mma_layout(
            self.w1_scale.reshape(num_experts_w1 * m1, k1_sf),
            m=m1,
            k=k1,
            num_groups=num_experts_w1,
        )

        assert self.w2_scale is not None
        num_experts_w2, m2, k2_sf = self.w2_scale.shape
        k2 = k2_sf * 16
        self.w2_sf_mma = flashinfer_convert_sf_to_mma_layout(
            self.w2_scale.reshape(num_experts_w2 * m2, k2_sf),
            m=m2,
            k=k2,
            num_groups=num_experts_w2,
        )

        # Eagerly initialize B12xMoEWrapper and pre-populate padded weights /
        # weight views during model loading so memory profiling accounts for
        # them and CUDA graph capture never runs lazy initialization.
        self._ensure_wrapper(device=layer.w13_weight.device)
        assert self._wrapper is not None
        dummy_x = torch.zeros(
            (1, self.hidden_dim),
            dtype=torch.bfloat16,
            device=layer.w13_weight.device,
        )
        dummy_ids = torch.zeros(
            (1, self.topk),
            dtype=torch.int32,
            device=layer.w13_weight.device,
        )
        dummy_weights = torch.full(
            (1, self.topk),
            1.0 / self.topk,
            dtype=torch.float32,
            device=layer.w13_weight.device,
        )
        self._wrapper.run(
            x=dummy_x,
            w1_weight=layer.w13_weight,
            w1_weight_sf=self.w1_sf_mma,
            w1_alpha=self.g1_alphas,
            fc2_input_scale=self._fc2_input_scale,
            w2_weight=layer.w2_weight,
            w2_weight_sf=self.w2_sf_mma,
            w2_alpha=self.g2_alphas,
            token_selected_experts=dummy_ids,
            token_final_scales=dummy_weights,
            input_global_scale=self._fc1_input_scale,
            output=dummy_x,
        )

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        p = current_platform
        return (
            p.is_cuda()
            and p.is_device_capability_family(120)
            and has_flashinfer_b12x_moe()
        )

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return True

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return (weight_key, activation_key) in (
            (kNvfp4Static, kNvfp4Dynamic),
            (kNvfp4Static, None),
        )

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in (
            MoEActivation.SILU,
            MoEActivation.GELU_TANH,
            MoEActivation.RELU2_NO_MUL,
        )

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        return not moe_parallel_config.use_ep

    def supports_expert_map(self) -> bool:
        return False

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        workspace1 = (1,)
        workspace2 = (0,)
        output_shape = (M, K)
        return (workspace1, workspace2, output_shape)

    @property
    def expects_unquantized_inputs(self) -> bool:
        return True

    def _ensure_wrapper(self, device: torch.device | str = "cuda") -> None:
        """Create B12xMoEWrapper sharing static/dynamic workspaces across layers."""
        if self._wrapper is not None:
            return

        from flashinfer.fused_moe import B12xMoEWrapper

        b12x_max_tokens = (
            B12X_MAX_DECODE_TOKENS
            if self._can_use_cutlass_prefill
            else self.max_num_tokens
        )
        resolved_device = torch.device(device)
        if resolved_device.type == "cuda" and resolved_device.index is None:
            resolved_device = torch.device("cuda", torch.cuda.current_device())

        ws_key = (
            resolved_device,
            self.global_num_experts,
            self.topk,
            self.hidden_dim,
            self.intermediate_size_per_partition,
            b12x_max_tokens,
            self.num_local_experts,
            self._activation_str,
        )
        shared_ws = self._SHARED_WORKSPACES.get(ws_key)
        if shared_ws is None:
            self._wrapper = B12xMoEWrapper(
                num_experts=self.global_num_experts,
                top_k=self.topk,
                hidden_size=self.hidden_dim,
                intermediate_size=self.intermediate_size_per_partition,
                use_cuda_graph=True,
                max_num_tokens=b12x_max_tokens,
                num_local_experts=self.num_local_experts,
                device=str(resolved_device),
                activation=self._activation_str,
            )
            self._SHARED_WORKSPACES[ws_key] = (
                self._wrapper._static_workspace,
                self._wrapper._dynamic_workspace,
                self._wrapper._moe_output,
            )
        else:
            static_ws, dynamic_ws, shared_out = shared_ws
            self._wrapper = B12xMoEWrapper(
                num_experts=self.global_num_experts,
                top_k=self.topk,
                hidden_size=self.hidden_dim,
                intermediate_size=self.intermediate_size_per_partition,
                use_cuda_graph=True,
                max_num_tokens=b12x_max_tokens,
                num_local_experts=self.num_local_experts,
                device=str(resolved_device),
                activation=self._activation_str,
                shared_static_workspace=static_ws,
                shared_dynamic_workspace=dynamic_ws,
                shared_output=shared_out,
            )

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor | None,
        workspace2: torch.Tensor | None,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool | None,
    ):
        num_tokens = hidden_states.shape[0]
        if (
            self._can_use_cutlass_prefill
            and num_tokens > B12X_MAX_DECODE_TOKENS
        ):
            assert self._cutlass_quant_scales is not None
            x_q, x_sf = scaled_fp4_quant(
                hidden_states, self.a1_gscale, is_sf_swizzled_layout=False
            )
            swiglu_limit = (
                self.gemm1_clamp_limit if activation == MoEActivation.SILU else None
            )
            flashinfer_cutlass_fused_moe(
                input=x_q,
                token_selected_experts=topk_ids.to(torch.int).contiguous(),
                token_final_scales=topk_weights,
                fc1_expert_weights=w1.view(torch.long),
                fc2_expert_weights=w2.view(torch.long),
                output_dtype=self.out_dtype,
                quant_scales=self._cutlass_quant_scales,
                input_sf=x_sf,
                fc1_expert_biases=None,
                fc2_expert_biases=None,
                swiglu_alpha=None,
                swiglu_beta=None,
                swiglu_limit=swiglu_limit,
                tp_size=1,
                tp_rank=0,
                ep_size=1,
                ep_rank=0,
                use_packed_weights=False,
                output=output,
                enable_pdl=None,
                use_mxfp8_act_scaling=False,
                use_w4_group_scaling=False,
                activation_type=self._cutlass_activation_type,
            )
            return

        assert self.w1_scale is not None and self.w2_scale is not None
        assert self.g1_alphas is not None and self.g2_alphas is not None
        assert self._fc2_input_scale is not None
        assert self.w1_sf_mma is not None and self.w2_sf_mma is not None

        self._ensure_wrapper(device=hidden_states.device)
        wrapper = self._wrapper
        assert wrapper is not None

        wrapper.run(
            x=hidden_states,
            w1_weight=w1,
            w1_weight_sf=self.w1_sf_mma,
            w1_alpha=self.g1_alphas,
            fc2_input_scale=self._fc2_input_scale,
            w2_weight=w2,
            w2_weight_sf=self.w2_sf_mma,
            w2_alpha=self.g2_alphas,
            token_selected_experts=topk_ids.to(torch.int32),
            token_final_scales=topk_weights,
            input_global_scale=self._fc1_input_scale,
            output=output,
        )
