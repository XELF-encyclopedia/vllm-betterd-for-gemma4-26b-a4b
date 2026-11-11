#!/usr/bin/env bash
# ==============================================================================
# Environment & GPU Verification Script for Blackwell SM120 Inference
# ==============================================================================
set -e

echo "=== System & GPU Diagnostic ==="
nvidia-smi --query-gpu=gpu_name,driver_version,memory.total --format=csv,noheader

echo -e "\n=== Checking CUDA & Torch Capabilities ==="
python3 -c "
import torch
print(f'PyTorch Version: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    props = torch.cuda.get_device_properties(0)
    print(f'Device Name: {props.name}')
    print(f'Compute Capability: SM{props.major}{props.minor}')
    print(f'Total Memory: {props.total_memory / (1024**3):.2f} GB')
    print(f'torch.float4_e2m1fn_x2 supported: {hasattr(torch, \"float4_e2m1fn_x2\")}')
"

echo -e "\n=== Checking vLLM & FlashInfer Installation ==="
python3 -c "
import vllm
print(f'vLLM version: {vllm.__version__}')
try:
    import flashinfer
    print(f'FlashInfer version: {flashinfer.__version__}')
except ImportError:
    print('[-] FlashInfer is NOT installed!')

from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import NvFp4MoeBackend
if hasattr(NvFp4MoeBackend, 'FLASHINFER_B12X'):
    print('[OK] FLASHINFER_B12X backend is registered in vLLM.')
else:
    print('[-] FLASHINFER_B12X backend is NOT registered. Please run apply_all_patches.sh.')
"
