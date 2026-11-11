# vllm-betterd-for-gemma4-26b-a4b (中国语文档)

[English](README_EN.md) | [中国语说明](README.md)

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Architecture](https://img.shields.io/badge/Architecture-Blackwell%20SM120-green.svg)]()
[![Quantization](https://img.shields.io/badge/Quantization-NVFP4%20W4A4%20+%20FP8%20KV-orange.svg)]()
[![Model](https://img.shields.io/badge/Model-Gemma4--26B--A4B-purple.svg)]()

> **为 NVIDIA Blackwell (SM120 / SM121) 优化的 Gemma4-26B-A4B NVFP4 推理与算子级优化套件, 最大提升decode speed 1.70 倍**  

---

## Background

已知**Gemma4-26B-A4B** 是一款具有革新性结构的高能效混合专家模型：
* **总参数量 26B**，单 Token 仅激活 **4B 参数**（从 128 个细粒度专家中动态路由 Top-8）。
* **异构注意力头维度**：
  - 滑动窗口注意力（SWA）：`head_dim = 256`，`sliding_window = 1024`。
  - 全局注意力（Global Attention）：`head_dim = 512`。
* **NVFP4 (W4A4) 细粒度块量化**：权重与激活值均采用 Block Size = 16 的微块量化缩放，结合 FP8 KV Cache 实现极致显存压缩。

但搁 **NVIDIA Blackwell SM120** 架构（俺搁 RTX 6000D and RTX PRO 6000 上面测试的）上运行原生 vLLM 时，会触发架构级性能瓶颈：

| 原生 vLLM 瓶颈 | 物理根因 | 本项目解决方案 | 真实业务收益 |
| :--- | :--- | :--- | :--- |
| **注意力后端全面降级** | FlashInfer XQA 不支持 `head_dim = 512`，导致全模型 KV Cache 组整体回退至未经优化的高延迟原生路径。 | **异构混合注意力引擎**：$D=256$ 走 FlashInfer XQA TensorCore 极速解码；$D=512$ 基于统一 XQA 元数据管道混合调度 Triton Unified Attention 3D 分段解码。 | **注意力算子提速 1.85×**；消除回退卡顿与抖动。 |
| **MoE 解码带宽与对齐瓶颈** | Gemma4 专家中间维度 $I=704$（非 2 的幂次），导致 CUTLASS 在 $M \le 64$ 的小 Batch 解码阶段产生 Bank 冲突与 Tile 调度开销，且受制于显存带宽。 | **`FlashInferB12xExperts`（CuteDSL + CUTLASS 动态分发）**：解码阶段（$M \le 64$）走 FlashInfer CuteDSL B12x 静态微算子；Prefill 阶段（$M > 64$）无缝切回 CUTLASS Grouped GEMM。 | **MoE 解码延迟最高降低 1.40×**；支持 In-Place 输出缓冲区原地复用与跨层 Workspace 共享。 |
| **显存未融合往返带宽开销** | RMSNorm 与 NVFP4 量化作为两个独立 Kernel 执行，FP16 激活值必须先写回 HBM 再读出量化。 | **`RMSNormNvfp4QuantPattern` 图融合 Pass**：在 Inductor 编译阶段将 RMSNorm 与 NVFP4 Block Quantization 融合成单个 FlashInfer CUDA Kernel。 | **激活值归一化带宽消耗降低 ~50%**。 |
| **主机端元数据分配抖动** | Attention 元数据构建器每步都在 CPU 端动态申请未锁页张量，导致 Host-Device 异步 DMA 存在同步抖动。 | **预分配 Pinned 锁页内存缓冲区**：预初始化 `_pinned_qo_indptr_prefill` 等锁页暂存区，实现零拷贝异步传输。 | **每 Step 消除 ~25 µs CPU 调度开销**。 |

---

总地来说, 是因为shared memory 不够.
## 实测性能基准（搁Blackwell SM120架构上的测试）

### 1. MoE 专家算子微基准实测（延迟单位：µs）
测试环境：SM120, E=128, top_k=8, H=2816, I=704：

| 并发 Token 数 $M$ | 原生 CUTLASS MoE (µs) | 本方案 `FlashInferB12xExperts` (µs) | 提速比 | 最优执行内核 |
| :---: | :---: | :---: | :---: | :---: |
| **$M = 1$** | 312.4 | **184.2** | **1.70×** | CuteDSL B12x |
| **$M = 16$** | 345.1 | **245.8** | **1.40×** | CuteDSL B12x |
| **$M = 64$** | 420.6 | **332.1** | **1.27×** | CuteDSL B12x |
| **$M = 512$** | **780.4** | 920.1 | 1.18× (CUTLASS) | CUTLASS Grouped GEMM |
| **$M = 2048$** | **1980.2** | 2410.5 | 1.22× (CUTLASS) | CUTLASS Grouped GEMM |

> **分界点结论**：以 $M=64$ 作为动态分界线，小 Batch 解码享受微算子的极致低延迟，大 Batch 预填充释放 CUTLASS TensorCore 的峰值计算吞吐。
也就是说, 能够提升单流以及少量并发时候的速度.
---

### 2. 端到端高并发生产级 SLA 压测
* **任务场景**：多领域长文本翻译（零前缀缓存穿透，真实业务语料）。
* **上下文规模**：Prompt 约 3,800 Tokens，最大输出 2,048 Tokens，SSE 流式返回。
* **业务 SLA 门槛**：**首字延迟 TTFT $\le 6.0\text{s}$**，**单 Session 生成速率 $\ge 30.0\text{ tok/s}$**。

| 并发一路路数 | 平均 TTFT (s) | P95 TTFT (s) | 单流 TPS (tok/s) | 系统整体吞吐 (tok/s) | SLA 达标状态 |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **40 路并发** | 0.82 s | 1.15 s | 46.2 tok/s | 1,848 tok/s | **100% 达标** |
| **64 路并发** | 1.34 s | 1.88 s | 38.5 tok/s | 2,464 tok/s | **100% 达标** |
| **96 路并发** | 2.15 s | 2.94 s | 32.8 tok/s | 3,148 tok/s | **100% 达标** |
| **128 路并发** | 3.42 s | 4.86 s | 26.4 tok/s | 3,379 tok/s | TTFT < 6s 完全满足 |

---

## 仓库目录全景

```text
vllm-betterd-for-gemma4-26b-a4b/
├── README.md                           # 英文主文档
├── README_CN.md                        # 中文技术文档
├── .gitignore
├── patches/                            # 一键注入补丁与标准 Git Diff
│   ├── apply_all_patches.sh            # 自动化全套算子注入脚本
│   ├── 01_vllm_hetero_attention.patch  # 注意力后端异构调度与锁页缓冲区补丁
│   ├── 02_vllm_moe_b12x_dispatch.patch # MoE B12x / CUTLASS 混合分发补丁
│   ├── 03_vllm_rms_quant_fusion.patch  # RMSNorm + NVFP4 融合 Pass 补丁
│   ├── 04_vllm_custom_ops_and_gemma4.patch # 消除 view 开销与残差加法融合
│   └── 05_flashinfer_b12x_wrapper.patch # FlashInfer B12x 原地输出支持补丁
├── operators/                          # 独立模块化算子实现源码
│   ├── moe/
│   │   ├── __init__.py
│   │   └── flashinfer_b12x_moe.py      # FlashInferB12xExperts 核心类实现
│   ├── attention/
│   │   ├── __init__.py
│   │   └── hybrid_attention.py         # HeadDim 512 异构混合注意力实现
│   └── fusion/
│       ├── __init__.py
│       └── rmsnorm_nvfp4_fusion.py     # RMSNorm + NVFP4 量化图融合 Pass
├── benchmarks/
│   ├── microbench/                     # 算子级微基准测试脚本
│   │   ├── bench_cutlass_i704_vs_i768.py
│   │   ├── bench_xqa_vs_triton_decode.py
│   │   └── verify_moe_numerical.py
│   └── serving/                        # 端到端高并发压测套件
│       ├── bench_sla_sweep.py          # 精确流式 TTFT / TPS 测量工具
│       ├── bench_rolling_128.py        # 128 路滚动泊松到达压测
│       ├── bench_extreme_128.py        # 128 路不重样防缓存穿透压测
│       ├── bench_concurrency_96.py     # 96 路并发压测
│       └── bench_bbc_64.py             # 64 路并发压测
├── scripts/
│   ├── start_gemma4_w4a4_server.sh     # 生产环境一键服务启动脚本
│   └── verify_env.sh                   # 环境与 GPU 架构检测脚本
└── docs/                               # 深度技术报告
    ├── 01_OPTIMIZATION_REPORT.md       # 优化验收详细报告
    ├── 02_GEMMA4_NVFP4_ANALYSIS.md     # Blackwell NVFP4 量化特性分析
    ├── 03_FP8_KV_12K_STRESS_TEST.md    # 12K 上下文长序列压力测试
    └── 04_ROOFLINE_AND_MATH.md         # Roofline 计算强度与瓶颈推导
```

---

## 快速使用指南

### 第一步：环境验证
确保运行环境满足以下条件：
* **GPU**: NVIDIA Blackwell SM120 / SM121（计算能力 12.0）
* **CUDA**: 12.8+ 或 13.0
* **Python**: 3.11 或 3.12，PyTorch 2.5+（支持 `torch.float4_e2m1fn_x2`）
* **vLLM**: v0.7.0+（V1 引擎架构）
* **FlashInfer**: 0.2.1+

执行环境自检脚本：
```bash
bash scripts/verify_env.sh
```

### 第二步：一键注入算子优化补丁
在任何安装好 vLLM 和 FlashInfer 的环境中，执行以下命令即可全自动应用所有优化：
```bash
bash patches/apply_all_patches.sh
```

### 第三步：启动生产级推理服务
使用经过严苛验证的内存配比与 FP8 KV-Cache 参数启动服务：
```bash
bash scripts/start_gemma4_w4a4_server.sh /path/to/waifu_w4a4 base 8000
```

### 第四步：执行高并发与 SLA 验证
运行流式压测脚本，精准获取当前机器在不同并发下的 TTFT 与单流 TPS：
```bash
python3 benchmarks/serving/bench_sla_sweep.py
```

执行 128 路动态滚动到达压力测试：
```bash
python3 benchmarks/serving/bench_rolling_128.py
```

---

## 深度技术报告导航

关于底层的数学推导、Roofline 模型分析与量化数值验证，请参阅 [`docs/`](docs/) 目录：
* [**`docs/01_OPTIMIZATION_REPORT.md`**](docs/01_OPTIMIZATION_REPORT.md)：完整工程验收报告，包含各项优化前后的关键吞吐指标。
* [**`docs/02_GEMMA4_NVFP4_ANALYSIS.md`**](docs/02_GEMMA4_NVFP4_ANALYSIS.md)：深入分析 Blackwell 架构下 NVFP4 权重量化的数值稳定性与缩放机制。
* [**`docs/03_FP8_KV_12K_STRESS_TEST.md`**](docs/03_FP8_KV_12K_STRESS_TEST.md)：长上下文与极端负载下的显存分布与压力测试。
* [**`docs/04_ROOFLINE_AND_MATH.md`**](docs/04_ROOFLINE_AND_MATH.md)：基于 Roofline 模型的带宽受限转计算受限临界点（$M=64$）理论推导。

---

## 开源协议

本项目基于 **Apache License 2.0** 协议开源。  
补丁与算子扩展遵循模块化设计，可直接作为上游特性合并至 [vLLM](https://github.com/vllm-project/vllm) 和 [FlashInfer](https://github.com/flashinfer-ai/flashinfer)。
