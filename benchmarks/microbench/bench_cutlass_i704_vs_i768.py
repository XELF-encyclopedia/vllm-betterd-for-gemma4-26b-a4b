#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Microbenchmark: CUTLASS Fused MoE Intermediate Size Alignment (I=704 vs I=768) on SM120.

Measures kernel execution latency across sequence lengths M in [64, 512, 2048, 8192, 15200]
to evaluate alignment overheads of non-power-of-two intermediate dimensions on Blackwell.
"""

import torch
from vllm._custom_ops import scaled_fp4_quant
from vllm.utils.flashinfer import flashinfer_cutlass_fused_moe

def run_bench():
    device = "cuda"
    E, H, top_k = 128, 2816, 8

    print(f"{'Intermediate':<14}{'Tokens (M)':<12}{'Latency (us)':<14}{'Status'}")
    print("-" * 50)

    for I in [704, 768]:
        w1 = torch.randint(0, 255, (E, 2 * I, H // 2), dtype=torch.uint8, device=device)
        w2 = torch.randint(0, 255, (E, H, I // 2), dtype=torch.uint8, device=device)
        w1_s = torch.ones((E, 2 * I, H // 16), dtype=torch.float8_e4m3fn, device=device)
        w2_s = torch.ones((E, H, I // 16), dtype=torch.float8_e4m3fn, device=device)
        g1_a = torch.full((E,), 0.001, dtype=torch.float32, device=device)
        g2_a = torch.full((E,), 0.001, dtype=torch.float32, device=device)
        a1_gs = torch.tensor(10.0, dtype=torch.float32, device=device)
        a2_gs = torch.tensor(10.0, dtype=torch.float32, device=device)
        quant_scales = [a1_gs, w1_s.view(torch.int32), g1_a, a2_gs, w2_s.view(torch.int32), g2_a]

        for M in [64, 512, 2048, 8192, 15200]:
            x = torch.randn((M, H), dtype=torch.bfloat16, device=device)
            out = torch.empty_like(x)
            topk_ids = torch.randint(0, E, (M, top_k), dtype=torch.int32, device=device)
            topk_weights = torch.full((M, top_k), 1.0 / top_k, dtype=torch.float32, device=device)

            def run_cutlass():
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
                    output=out,
                    activation_type=3,  # gelu_tanh
                )

            # Warmup
            for _ in range(10):
                run_cutlass()
            torch.cuda.synchronize()

            # Measure
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            iters = 30
            start.record()
            for _ in range(iters):
                run_cutlass()
            end.record()
            torch.cuda.synchronize()

            latency_us = start.elapsed_time(end) * 1000 / iters
            print(f"I={I:<12}{M:<12}{latency_us:<14.1f}PASS")

if __name__ == "__main__":
    run_bench()
