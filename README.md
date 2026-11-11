# vllm-betterd-for-gemma4-26b-a4b

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Architecture](https://img.shields.io/badge/Architecture-Blackwell%20SM120-green.svg)]()
[![Quantization](https://img.shields.io/badge/Quantization-NVFP4%20W4A4%20+%20FP8%20KV-orange.svg)]()
[![Model](https://img.shields.io/badge/Model-Gemma4--26B--A4B-purple.svg)]()

> **High-Throughput Blackwell (SM120) Operator & Inference Optimization Suite for Gemma4-26B-A4B NVFP4.**  

---

## Executive Summary & Motivation

**Gemma4-26B-A4B** represents a breakthrough architecture featuring:
* **26B Total Parameters**, with **4B Active Parameters** per token (Top-8 routing across 128 fine-grained experts).
* **Heterogeneous Attention Layers**:
  - Sliding Window Attention (SWA): `head_dim = 256`, `sliding_window = 1024`.
  - Global Attention: `head_dim = 512`.
* **NVFP4 (W4A4) Block Quantization**: Weight and activation block scales (block size = 16) with FP8 KV-Cache.

However, running Gemma4-26B-A4B on **NVIDIA Blackwell SM120** (RTX Pro 6000 Ada-Next / DGX Spark) under stock vLLM encounters severe architectural bottlenecks:

| Bottleneck in Stock vLLM | Root Cause | Our Solution in `vllm-betterd` | Real-World Impact |
| :--- | :--- | :--- | :--- |
| **Attention Backend Reversion** | FlashInfer XQA does not support `head_dim = 512`, reverting entire KV cache groups to native slow fallback. | **Heterogeneous Hybrid Attention Engine**: Routes $D=256$ to FlashInfer XQA TensorCore decode, and routes $D=512$ to Triton Unified Attention 3D decode using pre-built XQA metadata. | **1.85× Attention Speedup**; eliminates fallback latency spikes. |
| **MoE Decode Throughput Bound** | Non-power-of-two intermediate size ($I=704$) causes CUTLASS bank conflicts and tile setup overhead during small batch decode ($M \le 64$). | **`FlashInferB12xExperts` (CuteDSL + CUTLASS Hybrid Dispatch)**: Dispatches $M \le 64$ to FlashInfer CuteDSL B12x microkernel with static NVFP4 scaling; dynamically switches to CUTLASS for $M > 64$ prefill. | **Up to 1.40× MoE Decode Speedup**; zero-copy in-place output buffer reuse. |
| **Unfused Global Memory Traffic** | RMSNorm and NVFP4 block quantization execute as two distinct kernels, reading and writing FP16 activations to HBM twice. | **`RMSNormNvfp4QuantPattern` Graph Fusion Pass**: Compiles RMSNorm and NVFP4 block quantization into a single fused FlashInfer CUDA kernel launch. | **~50% Global Memory Bandwidth Reduction** on activation normalization. |
| **CPU DMA & Staging Jitter** | Metadata builder allocates unpinned CPU tensors every step, causing CUDA sync delays. | **Pre-allocated Pinned CPU Staging Buffers**: Pinned host buffers (`_pinned_qo_indptr_prefill`, etc.) ensure zero-copy async DMA to GPU. | **Eliminates ~25 µs CPU jitter** per decoding step. |

---

## Benchmark Highlights (Real-World SM120 Verification)

### 1. MoE Operator Microbenchmark (Latency in µs)
Evaluated on SM120 with $E=128, \text{top\_k}=8, H=2816, I=704$:

| Token Batch $M$ | Stock CUTLASS MoE (µs) | `FlashInferB12xExperts` (µs) | Speedup | Optimal Kernel |
| :---: | :---: | :---: | :---: | :---: |
| **$M = 1$** | 312.4 | **184.2** | **1.70×** | CuteDSL B12x |
| **$M = 16$** | 345.1 | **245.8** | **1.40×** | CuteDSL B12x |
| **$M = 64$** | 420.6 | **332.1** | **1.27×** | CuteDSL B12x |
| **$M = 512$** | **780.4** | 920.1 | 1.18× (CUTLASS) | CUTLASS Grouped GEMM |
| **$M = 2048$** | **1980.2** | 2410.5 | 1.22× (CUTLASS) | CUTLASS Grouped GEMM |

### 2. End-to-End High-Concurrency SLA Performance
* **Task**: Non-overlapping multi-domain technical translation (Zero KV cache reuse).
* **Context**: 3,800 Prompt Tokens, 2,048 Max Output Tokens, Streaming SSE with usage tracking.
* **SLA Thresholds**: $\text{TTFT} \le 6.0\text{s}$, $\text{Single-Session TPS} \ge 30.0\text{ tok/s}$.

| Concurrency Level | TTFT Avg (s) | P95 TTFT (s) | Single-Session TPS | Total System Throughput | SLA Compliance |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **40 Concurrent Streams** | 0.82 s | 1.15 s | 46.2 tok/s | 1,848 tok/s | **100% PASS** |
| **64 Concurrent Streams** | 1.34 s | 1.88 s | 38.5 tok/s | 2,464 tok/s | **100% PASS** |
| **96 Concurrent Streams** | 2.15 s | 2.94 s | 32.8 tok/s | 3,148 tok/s | **100% PASS** |
| **128 Concurrent Streams** | 3.42 s | 4.86 s | 26.4 tok/s | 3,379 tok/s | TTFT $\le 6$s met |

---

## Repository Structure

```text
vllm-betterd-for-gemma4-26b-a4b/
├── README.md                           # Comprehensive documentation & architecture guide
├── .gitignore
├── patches/                            # One-click installation & standard git diffs
│   ├── apply_all_patches.sh            # Automated bash patcher for vLLM & FlashInfer
│   ├── 01_vllm_hetero_attention.patch  # Unified diff: attention backend & pinned buffers
│   ├── 02_vllm_moe_b12x_dispatch.patch # Unified diff: MoE B12x / CUTLASS hybrid dispatch
│   ├── 03_vllm_rms_quant_fusion.patch  # Unified diff: RMSNorm + NVFP4 fusion pass
│   ├── 04_vllm_custom_ops_and_gemma4.patch # Unified diff: view avoidance & residual add fusion
│   └── 05_flashinfer_b12x_wrapper.patch # Unified diff: FlashInfer B12x inplace output support
├── operators/                          # Standalone modular operator implementations
│   ├── moe/
│   │   ├── __init__.py
│   │   └── flashinfer_b12x_moe.py      # FlashInferB12xExperts implementation
│   ├── attention/
│   │   ├── __init__.py
│   │   └── hybrid_attention.py         # Heterogeneous head_dim=512 hybrid attention
│   └── fusion/
│       ├── __init__.py
│       └── rmsnorm_nvfp4_fusion.py     # RMSNorm + NVFP4 FX fusion pass
├── benchmarks/
│   ├── microbench/                     # Low-level operator microbenchmarks
│   │   ├── bench_cutlass_i704_vs_i768.py
│   │   ├── bench_xqa_vs_triton_decode.py
│   │   └── verify_moe_numerical.py
│   └── serving/                        # High-concurrency end-to-end SLA stress suites
│       ├── bench_sla_sweep.py          # Accurate TTFT and decode TPS measurement
│       ├── bench_rolling_128.py        # 128-stream Poisson/rolling arrival test
│       ├── bench_extreme_128.py        # 128-stream non-overlapping translation
│       ├── bench_concurrency_96.py     # 96-stream non-overlapping translation
│       └── bench_bbc_64.py             # 64-stream non-overlapping translation
├── scripts/
│   ├── start_gemma4_w4a4_server.sh     # Production serving startup script
│   └── verify_env.sh                   # Environment diagnostic tool
└── docs/
    ├── 01_OPTIMIZATION_REPORT.md       # Full engineering acceptance report
    ├── 02_GEMMA4_NVFP4_ANALYSIS.md     # Blackwell NVFP4 + FP8 KV cache evaluation
    ├── 03_FP8_KV_12K_STRESS_TEST.md    # 12K context stress testing analysis
    └── 04_ROOFLINE_AND_MATH.md         # Roofline model & mathematical analysis
```

---

## Quick Start Guide

### Step 1: Environment Verification
Ensure your environment meets the prerequisites:
* **GPU**: NVIDIA Blackwell SM120 / SM121 (Compute Capability 12.0)
* **CUDA**: 12.8+ / 13.0
* **Python**: 3.11 or 3.12 with PyTorch 2.5+ (supporting `torch.float4_e2m1fn_x2`)
* **vLLM**: v0.7.0+ (V1 Engine)
* **FlashInfer**: 0.2.1+

Verify your local GPU environment:
```bash
bash scripts/verify_env.sh
```

### Step 2: Apply Optimization Patches
Apply the full suite of operator optimizations to your installed `vllm` and `flashinfer` packages with a single command:
```bash
bash patches/apply_all_patches.sh
```

### Step 3: Launch Production Serving
Start the optimized vLLM server with calibrated memory utilization and FP8 KV-Cache:
```bash
bash scripts/start_gemma4_w4a4_server.sh /path/to/waifu_w4a4 base 8000
```

### Step 4: Run SLA & Concurrency Verification
Run the streaming benchmark to verify TTFT and single-session TPS:
```bash
python3 benchmarks/serving/bench_sla_sweep.py
```

To run the extreme 128-stream continuous rolling arrival test:
```bash
python3 benchmarks/serving/bench_rolling_128.py
```

---

## 📖 Deep Technical Reports

For detailed theoretical derivations, roofline models, and empirical data, refer to our documents in [`docs/`](docs/):
* [**`docs/01_OPTIMIZATION_REPORT.md`**](docs/01_OPTIMIZATION_REPORT.md): Detailed acceptance test results and throughput comparisons.
* [**`docs/02_GEMMA4_NVFP4_ANALYSIS.md`**](docs/02_GEMMA4_NVFP4_ANALYSIS.md): Comprehensive analysis of NVFP4 W4A4 quantization stability.
* [**`docs/03_FP8_KV_12K_STRESS_TEST.md`**](docs/03_FP8_KV_12K_STRESS_TEST.md): Long-context performance and memory footprint analysis.
* [**`docs/04_ROOFLINE_AND_MATH.md`**](docs/04_ROOFLINE_AND_MATH.md): Mathematical formulation of the memory-bound to compute-bound crossover point ($M=64$).

---

## License & Attribution

This project is licensed under the **Apache License 2.0**.  
Patches and operator extensions are designed for seamless upstream integration into [vLLM](https://github.com/vllm-project/vllm) and [FlashInfer](https://github.com/flashinfer-ai/flashinfer).
