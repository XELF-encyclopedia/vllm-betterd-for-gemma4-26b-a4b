#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Microbenchmark: Numerical Accuracy and Consistency Verification for B12x CuteDSL vs CUTLASS MoE.

Validates that both CuteDSL micro-kernel and CUTLASS grouped GEMM produce mathematically
equivalent outputs (within float8/float4 quantization bounds) using synthetic or real weights.
"""

import torch
from vllm._custom_ops import scaled_fp4_quant
from vllm.utils.flashinfer import flashinfer_cutlass_fused_moe, flashinfer_convert_sf_to_mma_layout
from flashinfer.fused_moe import B12xMoEWrapper

def verify_moe():
    device = "cuda:0"
    E, H, I, top_k = 128, 2816, 704, 8
    M = 16

    print("Initializing synthetic NVFP4 MoE weights...")
    w1 = torch.randint(0, 255, (E, 2 * I, H // 2), dtype=torch.uint8, device=device)
    w2 = torch.randint(0, 255, (E, H, I // 2), dtype=torch.uint8, device=device)
    w1_s = torch.ones((E, 2 * I, H // 16), dtype=torch.float8_e4m3fn, device=device)
    w2_s = torch.ones((E, H, I // 16), dtype=torch.float8_e4m3fn, device=device)
    g1_a = torch.full((E,), 0.001, dtype=torch.float32, device=device)
    g2_a = torch.full((E,), 0.001, dtype=torch.float32, device=device)
    a1_gs = torch.tensor(10.0, dtype=torch.float32, device=device)
    a2_gs = torch.tensor(10.0, dtype=torch.float32, device=device)
    quant_scales = [a1_gs, w1_s.view(torch.int32), g1_a, a2_gs, w2_s.view(torch.int32), g2_a]

    x = torch.randn((M, H), dtype=torch.bfloat16, device=device)
    topk_ids = torch.randint(0, E, (M, top_k), dtype=torch.int32, device=device)
    topk_weights = torch.full((M, top_k), 1.0 / top_k, dtype=torch.float32, device=device)

    # CUTLASS output
    out_cutlass = torch.empty_like(x)
    x_q, x_sf = scaled_fp4_quant(x, a1_gs, is_sf_swizzled_layout=False)
    flashinfer_cutlass_fused_moe(
        input=x_q,
        token_selected_experts=topk_ids,
        token_final_scales=topk_weights,
        fc1_expert_weights=w1.view(torch.long),
        fc2_expert_weights=w2.view(torch.long),
        output_dtype=torch.bfloat16,
        quant_scales=quant_scales,
        input_sf=x_sf,
        output=out_cutlass,
        activation_type=3,  # gelu_tanh
    )

    # CuteDSL B12x output
    w1_sf_mma = flashinfer_convert_sf_to_mma_layout(
        w1_s.reshape(E * 2 * I, H // 16), m=2 * I, k=H, num_groups=E
    )
    w2_sf_mma = flashinfer_convert_sf_to_mma_layout(
        w2_s.reshape(E * H, I // 16), m=H, k=I, num_groups=E
    )
    wrapper = B12xMoEWrapper(
        num_experts=E, top_k=top_k, hidden_size=H, intermediate_size=I,
        use_cuda_graph=False, max_num_tokens=64, num_local_experts=E,
        device=device, activation="gelu_tanh"
    )
    out_b12x = torch.empty_like(x)
    wrapper.run(
        x=x,
        w1_weight=w1,
        w1_weight_sf=w1_sf_mma,
        w1_alpha=g1_a,
        fc2_input_scale=torch.ones(E, device=device, dtype=torch.float32),
        w2_weight=w2,
        w2_weight_sf=w2_sf_mma,
        w2_alpha=g2_a,
        token_selected_experts=topk_ids,
        token_final_scales=topk_weights,
        input_global_scale=(1.0 / a1_gs).to(torch.float32),
        output=out_b12x,
    )

    diff = (out_cutlass.float() - out_b12x.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    cos_sim = torch.nn.functional.cosine_similarity(
        out_cutlass.flatten().float().unsqueeze(0),
        out_b12x.flatten().float().unsqueeze(0)
    ).item()

    print(f"Cosine Similarity: {cos_sim:.5f}")
    print(f"Max Absolute Error: {max_diff:.5f}")
    print(f"Mean Absolute Error: {mean_diff:.5f}")
    if cos_sim > 0.98:
        print("[OK] Numerical accuracy verification passed!")
    else:
        print("[WARNING] Numerical divergence detected, check scales.")

if __name__ == "__main__":
    verify_moe()
