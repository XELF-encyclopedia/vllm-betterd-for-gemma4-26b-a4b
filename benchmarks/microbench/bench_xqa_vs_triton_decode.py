#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Microbenchmark: Heterogeneous Attention Decode Latency on SM120.

Measures:
1. FlashInfer XQA TensorCore Decode at head_dim=256 (Sliding Window Attention layers).
2. Triton Unified Attention 3D Segmented Decode at head_dim=512 (Global Attention layers).
Across batch sizes B in [1, 4, 8, 16, 32, 64].
"""

import torch
from vllm.v1.attention.ops.triton_unified_attention import unified_attention
from vllm.v1.kv_cache_interface import KVQuantMode
from vllm.utils.flashinfer import flashinfer_xqa_batch_decode_with_kv_cache

def bench_triton_head_dim_512():
    device = "cuda"
    block_size = 16
    num_blocks = 4096
    nq, nkv, hs = 16, 2, 512

    kv_cache = torch.randint(0, 100, (num_blocks, nkv, block_size, 2 * hs), dtype=torch.uint8, device=device).view(torch.float8_e4m3fn)
    kv_perm = kv_cache.transpose(1, 2)
    k_cache, v_cache = kv_perm.split(hs, dim=-1)

    k_scale = torch.tensor([1.0], dtype=torch.float32, device=device)
    v_scale = torch.tensor([1.0], dtype=torch.float32, device=device)

    max_tokens_3d = 64
    segm_out = torch.empty((max_tokens_3d, nq, 64, 512), dtype=torch.float32, device=device)
    segm_max = torch.empty((max_tokens_3d, nq, 64), dtype=torch.float32, device=device)
    segm_exp = torch.empty((max_tokens_3d, nq, 64), dtype=torch.float32, device=device)

    print("\n--- Triton Unified Attention (head_dim=512, Global Attention) ---")
    for B in [1, 4, 8, 16, 32, 64]:
        seq_len = 3800
        q = torch.randn((B, nq, hs), dtype=torch.bfloat16, device=device)
        out = torch.empty_like(q)
        cu_q = torch.arange(0, B + 1, dtype=torch.int32, device=device)
        seq_lens = torch.full((B,), seq_len, dtype=torch.int32, device=device)
        max_blocks = (seq_len + block_size - 1) // block_size
        block_table = torch.randint(0, num_blocks, (B, max_blocks), dtype=torch.int32, device=device)
        k_descale = k_scale.expand(B, nkv)
        v_descale = v_scale.expand(B, nkv)
        segs = 64 if B == 1 else 16
        s_out = segm_out if segs == 64 else segm_out.view(-1, nq, 16, 512)
        s_max = segm_max if segs == 64 else segm_max.view(-1, nq, 16)
        s_exp = segm_exp if segs == 64 else segm_exp.view(-1, nq, 16)

        def run_triton():
            unified_attention(
                q=q, k=k_cache, v=v_cache, out=out,
                cu_seqlens_q=cu_q, max_seqlen_q=1,
                seqused_k=seq_lens, max_seqlen_k=seq_len,
                softmax_scale=1.0, causal=True,
                window_size=(-1, -1), block_table=block_table,
                softcap=0, q_descale=None, k_descale=k_descale, v_descale=v_descale,
                seq_threshold_3D=64, num_par_softmax_segments=segs,
                softmax_segm_output=s_out, softmax_segm_max=s_max, softmax_segm_expsum=s_exp,
                kv_quant_mode=KVQuantMode.FP8_PER_TENSOR,
            )

        for _ in range(10):
            run_triton()
        torch.cuda.synchronize()

        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        iters = 50
        s.record()
        for _ in range(iters):
            run_triton()
        e.record()
        torch.cuda.synchronize()
        print(f"Batch={B:2d}: Latency = {s.elapsed_time(e)*1000/iters:6.1f} us")

def bench_xqa_head_dim_256():
    device = "cuda"
    H_q, H_kv, D = 16, 8, 256
    page_size = 16
    seq_len = 2048
    pages_per_seq = seq_len // page_size
    num_pages = 64 * pages_per_seq + 16
    workspace = torch.zeros(256 * 1024 * 1024, dtype=torch.uint8, device=device)
    k_cache = torch.randn(num_pages, page_size, H_kv, D, dtype=torch.bfloat16, device=device).to(torch.float8_e4m3fn)
    v_cache = torch.randn(num_pages, page_size, H_kv, D, dtype=torch.bfloat16, device=device).to(torch.float8_e4m3fn)

    print("\n--- FlashInfer XQA TensorCore (head_dim=256, Sliding Window Attention) ---")
    for B in [4, 8, 16, 32, 64]:
        q = torch.randn(B, H_q, D, dtype=torch.bfloat16, device=device)
        out = torch.empty_like(q)
        block_tables = torch.arange(0, B * pages_per_seq, dtype=torch.int32, device=device).view(B, pages_per_seq)
        seq_lens = torch.full((B,), seq_len, dtype=torch.int32, device=device)

        def run_xqa():
            flashinfer_xqa_batch_decode_with_kv_cache(
                query=q,
                kv_cache=(k_cache, v_cache),
                kv_cache_sf=None,
                workspace_buffer=workspace,
                block_tables=block_tables,
                seq_lens=seq_lens,
                max_seq_len=seq_len,
                bmm1_scale=D**-0.5,
                bmm2_scale=1.0,
                window_left=1023,
                out=out,
                sinks=None,
                kv_layout="NHD",
                q_len_per_req=1,
            )

        for _ in range(5):
            run_xqa()
        torch.cuda.synchronize()

        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        iters = 50
        s.record()
        for _ in range(iters):
            run_xqa()
        e.record()
        torch.cuda.synchronize()
        print(f"Batch={B:2d}: Latency = {s.elapsed_time(e)*1000/iters:6.1f} us")

if __name__ == "__main__":
    bench_triton_head_dim_512()
    bench_xqa_head_dim_256()
