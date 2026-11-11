#!/usr/bin/env bash
# ==============================================================================
# Production Startup Script: Gemma4-26B-A4B NVFP4 on Blackwell SM120 (vLLM v1)
# ==============================================================================
set -e

MODEL_PATH=${1:-"/root/autodl-tmp/models/waifu_w4a4"}
SERVED_NAME=${2:-"base"}
PORT=${3:-8000}

echo "=== Starting Gemma4-26B-A4B NVFP4 Server on Port $PORT ==="
echo "Model Path: $MODEL_PATH"
echo "Served Model Name: $SERVED_NAME"

export VLLM_USE_V1=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Ensure flashinfer is selected for SM120 heterogeneous attention
export VLLM_ATTENTION_BACKEND=FLASHINFER

LOG_FILE="vllm_gemma4_sm120.log"
: > "$LOG_FILE"

pkill -f 'vllm.entrypoints.openai.api_server' 2>/dev/null || true
sleep 3

python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --served-model-name "$SERVED_NAME" \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port "$PORT" \
  --max-model-len 32768 \
  --max-num-seqs 128 \
  --max-num-batched-tokens 32768 \
  --gpu-memory-utilization 0.88 \
  --kv-cache-dtype fp8 \
  >> "$LOG_FILE" 2>&1 &

PID=$!
echo "Server spawned with PID $PID. Tailing logs to $LOG_FILE..."

for i in $(seq 1 60); do
  sleep 3
  if curl -s -m 2 -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -q 200; then
    echo "[OK] Gemma4-26B-A4B vLLM server is healthy and ready!"
    break
  fi
  if ! kill -0 $PID 2>/dev/null; then
    echo "[-] Server exited prematurely. Last 20 lines of log:"
    tail -n 20 "$LOG_FILE"
    exit 1
  fi
  printf "."
done
