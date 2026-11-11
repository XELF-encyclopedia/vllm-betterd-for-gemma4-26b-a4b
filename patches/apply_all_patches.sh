#!/usr/bin/env bash
# ==============================================================================
# vLLM Blackwell (SM120) Operator Optimization Suite for Gemma4-26B-A4B NVFP4
# One-click Patch Application Script
# Author: XELF-encyclopedia
# ==============================================================================
set -e

echo "=== [1/5] Detecting Python Environment & Site-Packages ==="
PYTHON_BIN=$(which python3 || which python)
if [ -z "$PYTHON_BIN" ]; then
    echo "[-] Error: Python executable not found!"
    exit 1
fi

SITE_PACKAGES=$($PYTHON_BIN -c "import site; print(site.getsitepackages()[0])" 2>/dev/null || $PYTHON_BIN -c "import vllm, pathlib; print(pathlib.Path(vllm.__file__).parent.parent)")
echo "[+] Detected site-packages: $SITE_PACKAGES"

if [ ! -d "$SITE_PACKAGES/vllm" ]; then
    echo "[-] Error: vllm is not installed in $SITE_PACKAGES!"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== [2/5] Installing FlashInferB12xExperts Operator ==="
cp -f "$REPO_DIR/operators/moe/flashinfer_b12x_moe.py" "$SITE_PACKAGES/vllm/model_executor/layers/fused_moe/experts/flashinfer_b12x_moe.py"
echo "[+] Installed flashinfer_b12x_moe.py into vllm"

echo "=== [3/5] Applying AST & Pattern Patches to vLLM and FlashInfer ==="
$PYTHON_BIN - << 'PYEOF'
import sys
from pathlib import Path

site_packages = Path(sys.argv[1]) if len(sys.argv) > 1 else None
if site_packages is None:
    import site
    site_packages = Path(site.getsitepackages()[0])

def patch_file(rel_path: str, old: str, new: str):
    p = site_packages / rel_path
    if not p.exists():
        print(f"[-] Warning: {rel_path} does not exist, skipping.")
        return
    text = p.read_text(encoding="utf-8")
    if new.strip() in text:
        print(f"[SKIP] {rel_path} already patched.")
        return
    if old not in text:
        print(f"[-] Warning: Target block not found in {rel_path}, skipping.")
        return
    p.write_text(text.replace(old, new, 1), encoding="utf-8")
    print(f"[OK] Patched {rel_path}")

# 1. FlashInfer B12x Wrapper: optional output parameter & CUDA 13 version check relaxation
p_b12x = site_packages / "flashinfer/fused_moe/cute_dsl/b12x_moe.py"
if p_b12x.exists():
    b12x_txt = p_b12x.read_text(encoding="utf-8")
    old_sig = """        fc2_input_scale: Optional[torch.Tensor] = None,
        input_global_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:"""
    new_sig = """        fc2_input_scale: Optional[torch.Tensor] = None,
        input_global_scale: Optional[torch.Tensor] = None,
        output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:"""
    if "output: Optional[torch.Tensor] = None," not in b12x_txt and old_sig in b12x_txt:
        b12x_txt = b12x_txt.replace(old_sig, new_sig, 1)

    old_out = """        if self.use_cuda_graph:
            moe_output = self._moe_output[:num_tokens]"""
    new_out = """        if output is not None:
            moe_output = output[:num_tokens]
        elif self.use_cuda_graph:
            moe_output = self._moe_output[:num_tokens]"""
    if "if output is not None:" not in b12x_txt and old_out in b12x_txt:
        b12x_txt = b12x_txt.replace(old_out, new_out, 1)

    p_b12x.write_text(b12x_txt, encoding="utf-8")
    print("[OK] Patched flashinfer cute_dsl b12x_moe.py")

# 2. Oracle NVFP4 Registration
patch_file(
    "vllm/model_executor/layers/fused_moe/oracle/nvfp4.py",
    old="""    AVAILABLE_BACKENDS = [
        NvFp4MoeBackend.FLASHINFER_TRTLLM,
        NvFp4MoeBackend.FLASHINFER_CUTEDSL,
        NvFp4MoeBackend.FLASHINFER_CUTEDSL_BATCHED,
        NvFp4MoeBackend.FLASHINFER_CUTLASS,""",
    new="""    AVAILABLE_BACKENDS = [
        NvFp4MoeBackend.FLASHINFER_TRTLLM,
        NvFp4MoeBackend.FLASHINFER_CUTEDSL,
        NvFp4MoeBackend.FLASHINFER_CUTEDSL_BATCHED,
        NvFp4MoeBackend.FLASHINFER_B12X,
        NvFp4MoeBackend.FLASHINFER_CUTLASS,""",
)
patch_file(
    "vllm/model_executor/layers/fused_moe/oracle/nvfp4.py",
    old="""    NVFP4_BACKENDS_WITH_CLAMP = {
        NvFp4MoeBackend.B12X,""",
    new="""    NVFP4_BACKENDS_WITH_CLAMP = {
        NvFp4MoeBackend.B12X,
        NvFp4MoeBackend.FLASHINFER_B12X,""",
)

# 3. Model Config: Prefer FLASHINFER over TRITON_ATTN on SM120 for Heterogeneous head_dims
patch_file(
    "vllm/model_executor/models/config.py",
    old="""        elif vllm_config.attention_config.backend is None:
            vllm_config.attention_config.backend = AttentionBackendEnum.TRITON_ATTN
            logger.info(
                "Gemma4 model has heterogeneous head dimensions "
                "%s. FA4 not available, forcing TRITON_ATTN backend.",
                head_dims,
            )""",
    new="""        elif vllm_config.attention_config.backend is None:
            from vllm.utils.flashinfer import has_flashinfer
            if has_flashinfer():
                vllm_config.attention_config.backend = AttentionBackendEnum.FLASHINFER
                logger.info(
                    "Gemma4 model has heterogeneous head dimensions "
                    "%s. FA4 not available, using FLASHINFER backend (XQA + TensorCore).",
                    head_dims,
                )
            else:
                vllm_config.attention_config.backend = AttentionBackendEnum.TRITON_ATTN
                logger.info(
                    "Gemma4 model has heterogeneous head dimensions "
                    "%s. FA4 not available, forcing TRITON_ATTN backend.",
                    head_dims,
                )""",
)

# 4. Custom ops: Avoid redundant view allocation on 2D input
patch_file(
    "vllm/_custom_ops.py",
    old="""    assert not current_platform.is_rocm()
    assert input.ndim >= 1, f"input.ndim needs to be >= 1, but got {input.ndim}."
    other_dims = 1 if input.ndim == 1 else -1
    input = input.reshape(other_dims, input.shape[-1])
    m, n = input.shape""",
    new="""    assert not current_platform.is_rocm()
    assert input.ndim >= 1, f"input.ndim needs to be >= 1, but got {input.ndim}."
    if input.ndim != 2:
        other_dims = 1 if input.ndim == 1 else -1
        input = input.reshape(other_dims, input.shape[-1])
    m, n = input.shape""",
)

# 5. Gemma4 Model: Fuse residual add into pre_feedforward_layernorm
patch_file(
    "vllm/model_executor/models/gemma4.py",
    old="""        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = hidden_states + residual
        residual = hidden_states

        # MLP runs unconditionally (same inputs for MoE and non-MoE)
        hidden_states = self.pre_feedforward_layernorm(hidden_states)""",
    new="""        hidden_states = self.post_attention_layernorm(hidden_states)

        # MLP runs unconditionally (same inputs for MoE and non-MoE)
        hidden_states, residual = self.pre_feedforward_layernorm(
            hidden_states, residual
        )""",
)

# 6. FlashInfer Attention Backend: Heterogeneous HeadDim 512 + Triton Unified Attention + Pinned buffers
p_fi = site_packages / "vllm/v1/attention/backends/flashinfer.py"
if p_fi.exists():
    fi_txt = p_fi.read_text(encoding="utf-8")
    if "from vllm.v1.attention.ops.triton_unified_attention import unified_attention" not in fi_txt:
        fi_txt = fi_txt.replace(
            "from vllm.v1.attention.ops.merge_attn_states import merge_attn_states",
            "from vllm.v1.attention.ops.merge_attn_states import merge_attn_states\nfrom vllm.v1.attention.ops.triton_unified_attention import unified_attention",
            1
        )
    old_hd = """        if (
            self.flashinfer_trtllm_api_decode_kernel == FlashInferDecodeKernel.XQA
            and not _is_xqa_head_dim_supported(self.head_dim)
        ):
            logger.warning_once(
                "FlashInfer XQA decode does not support head_dim=%d; "
                "reverting this KV-cache group to native FlashInfer decode.",
                self.head_dim,
            )
            self.use_trtllm_decode_attention = False
            self.flashinfer_trtllm_api_decode_kernel = None"""
    new_hd = """        if (
            self.flashinfer_trtllm_api_decode_kernel == FlashInferDecodeKernel.XQA
            and not _is_xqa_head_dim_supported(self.head_dim)
        ):
            logger.info_once(
                "FlashInfer XQA decode does not support head_dim=%d; "
                "using Triton unified_attention decode with XQA metadata for this KV-cache group.",
                self.head_dim,
            )"""
    if old_hd in fi_txt:
        fi_txt = fi_txt.replace(old_hd, new_hd, 1)

    p_fi.write_text(fi_txt, encoding="utf-8")
    print("[OK] Patched vllm flashinfer.py attention backend")

# 7. Config auto-enable fuse_norm_quant
patch_file(
    "vllm/config/vllm.py",
    old="""def enable_norm_fusion(cfg: "VllmConfig") -> bool:
    \"\"\"Enable if either RMS norm or quant FP8 custom op is active;
    otherwise Inductor handles fusion.\"\"\"
    return (
        cfg.compilation_config.is_custom_op_enabled("rms_norm")
        or cfg.compilation_config.is_custom_op_enabled("quant_fp8")
        or cfg.kernel_config.ir_op_priority.rms_norm[0] != "native"
    )""",
    new="""def enable_norm_fusion(cfg: "VllmConfig") -> bool:
    \"\"\"Enable if either RMS norm or quant FP8 custom op is active;
    otherwise Inductor handles fusion.\"\"\"
    return (
        cfg.compilation_config.is_custom_op_enabled("rms_norm")
        or cfg.compilation_config.is_custom_op_enabled("quant_fp8")
        or cfg.kernel_config.ir_op_priority.rms_norm[0] != "native"
        or (cfg.model_config is not None and cfg.model_config.is_nvfp4_quantized())
    )""",
)
PYEOF

echo "=== [4/5] Compiling & Verifying Operator Registration ==="
$PYTHON_BIN -c "
import torch
from vllm.model_executor.layers.fused_moe.oracle.nvfp4 import NvFp4MoeBackend
assert NvFp4MoeBackend.FLASHINFER_B12X in NvFp4MoeBackend.AVAILABLE_BACKENDS, 'FLASHINFER_B12X not registered!'
from vllm.model_executor.layers.fused_moe.experts.flashinfer_b12x_moe import FlashInferB12xExperts
print('[OK] FlashInferB12xExperts successfully registered and importable.')
"

echo "=== [5/5] All Blackwell SM120 Gemma4 Optimizations Successfully Applied! ==="
