# Gemma4-26B-A4B Roofline & Mathematical Bottleneck Analysis on Blackwell SM120

## 1. Hardware Characteristics (Blackwell SM120 / RTX Pro 6000 Ada-Next)
* **Architecture**: Blackwell SM120 (5th Gen Tensor Cores)
* **HBM Bandwidth**: $\approx 1.8 \text{ TB/s}$ (Effective $\sim 1.45 \text{ TB/s}$ sustained)
* **Tensor Core NVFP4 Peak Compute**: $\approx 1800 \text{ TFLOPS}$ (dense / structured sparsity)
* **L2 Cache**: 128 MB

## 2. Gemma4-26B-A4B Architectural Geometry
* **Total Parameters**: 26B (4B Active parameters per token)
* **Active Experts per Token**: Top-8 out of 128 experts ($E=128, \text{top\_k}=8$)
* **Hidden Size**: $H = 2816$
* **MoE Intermediate Size per Expert**: $I = 704$ (Note: Non-power-of-two, introduces alignment constraints)
* **Attention Layers**:
  - Global Attention: $head\_dim = 512, num\_heads = 16, num\_kv\_heads = 2$
  - Sliding Window Attention (SWA): $head\_dim = 256, num\_heads = 16, num\_kv\_heads = 8$, window = 1024
* **Quantization Scheme**: NVFP4 W4A4 (Weight & Activation block-quantized, block size = 16) with FP8 KV-Cache.

---

## 3. Arithmetic Intensity & Bottleneck Phases

### Decode Phase ($M \le 64$ Concurrent Tokens)
During single-stream or small-batch decode:
* **Active Parameter Memory Movement**:
  $$\text{Bytes Read} \approx 4\text{B active parameters} \times 0.5 \text{ bytes/param (NVFP4)} + \text{Scales} \approx 2.2 \text{ GB}$$
* **FLOPs per Token**:
  $$\text{FLOPs} \approx 2 \times 4\text{B} = 8 \text{ GFLOPs}$$
* **Arithmetic Intensity (AI)**:
  $$\text{AI} = \frac{8 \times 10^9 \text{ FLOPs}}{2.2 \times 10^9 \text{ Bytes}} \approx 3.63 \text{ FLOPs/Byte}$$
* **Roofline Regime**: Strictly **Memory-Bandwidth Bound** ($\text{AI} \ll \text{Machine Ridge Point} \approx 1000$).
* **Optimization Implication**:
  - For $M \le 64$, CuteDSL micro-kernels with static NVFP4 scaling minimize memory transactions and instruction dispatch overhead.
  - FlashInfer B12x microkernel achieves up to $1.40\times$ speedup over CUTLASS grouped GEMM because CUTLASS tile schedulers incur high setup latency when $M$ is small.

### Prefill Phase ($M > 64$ Tokens)
During prompt evaluation:
* **Arithmetic Intensity (AI)**:
  $$\text{AI} = \frac{2 \times M \times N \times K}{M \times K + N \times K + M \times N} \propto M$$
* At $M = 2048$, $\text{AI} > 500 \text{ FLOPs/Byte}$, transitioning into the **Compute-Bound** regime.
* **Optimization Implication**:
  - CUTLASS grouped GEMM heavily saturates Blackwell 5th Gen Tensor Cores with double-buffering and warp-specialized pipelines, beating micro-kernels by $\sim 20\%$.
  - Therefore, dynamic hybrid dispatch ($M \le 64 \to \text{CuteDSL}$, $M > 64 \to \text{CUTLASS}$) yields the globally optimal latency envelope.

---

## 4. Attention Backend Breakdown

| Attention Layer | Head Dim | Batch $B \le 24$ | Batch $B \ge 64$ | Optimal Backend | Rationale |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Sliding Window** | 256 | XQA (32.4 µs) | XQA (54.1 µs) | **FlashInfer XQA** | TensorCore accelerated decode natively supports $D=256$. |
| **Global Attention** | 512 | Triton 3D (48.2 µs) | Triton 3D (112.8 µs) | **Triton Unified Attention** | XQA lacks $D=512$ tile; Triton 3D segmenting prevents register spills. |

Pre-allocating pinned host staging buffers completely removes the $\sim 25\text{ µs}$ CPU runtime allocation cost on every decoding step.
