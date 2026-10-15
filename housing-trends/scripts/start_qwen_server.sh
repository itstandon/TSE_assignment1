#!/usr/bin/env bash
# Serve the local Qwen 2.5 7B Instruct (GGUF, Q4_K_M) with llama.cpp's llama-server: an OpenAI-compatible API at
# http://$QWEN_HOST:$QWEN_PORT/v1, which LLM_PROVIDER=qwen (the default in housing_trends/llm.py) talks to.
# Run it on a GPU node (the llama-server builds need CUDA), e.g. inside an interactive job:
#   srun -p u22 --gres=gpu:1 --pty bash      then    bash scripts/start_qwen_server.sh
# Foreground process: stop it with Ctrl-C. Every setting can be overridden through the environment:
#   LLAMA_SERVER  path to the llama-server binary      QWEN_GGUF  first shard of the model
#   QWEN_HOST     bind address (default 127.0.0.1)     QWEN_PORT  port (default 8081; 8080 is used by vrspecml)
#   QWEN_CTX      context size (default 8192)          QWEN_NGL   layers offloaded to the GPU (default 99 = all)
set -euo pipefail

QWEN_GGUF="${QWEN_GGUF:-$HOME/llm-models/qwen2.5-7b-instruct/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf}"
QWEN_HOST="${QWEN_HOST:-127.0.0.1}"
QWEN_PORT="${QWEN_PORT:-8081}"
QWEN_CTX="${QWEN_CTX:-8192}"
QWEN_NGL="${QWEN_NGL:-99}"

if [[ -z "${LLAMA_SERVER:-}" ]]; then
    for c in "$(command -v llama-server || true)" "$HOME/venvs/vrspecml/bin/llama-server" \
             "$HOME/build/llama.cpp/build-cuda/bin/llama-server"; do
        if [[ -n "$c" && -x "$c" ]]; then LLAMA_SERVER="$c"; break; fi
    done
fi
[[ -n "${LLAMA_SERVER:-}" ]] || { echo "llama-server not found; set LLAMA_SERVER=/path/to/llama-server" >&2; exit 1; }
[[ -f "$QWEN_GGUF" ]] || { echo "model not found: $QWEN_GGUF (set QWEN_GGUF)" >&2; exit 1; }

echo "serving $QWEN_GGUF with $LLAMA_SERVER at http://$QWEN_HOST:$QWEN_PORT/v1"
exec "$LLAMA_SERVER" -m "$QWEN_GGUF" --alias qwen2.5-7b-instruct --host "$QWEN_HOST" --port "$QWEN_PORT" \
     -c "$QWEN_CTX" -ngl "$QWEN_NGL" --seed 42
