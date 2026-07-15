#!/bin/bash
set -euo pipefail

# 环境与工作目录
source "$(dirname "$0")/../../set_env.sh"
cd "$(dirname "$0")"

export ASCEND_RT_VISIBLE_DEVICES=4

# 带时间戳的日志
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "=== Accuracy test start ==="
python -u run_test.py --mode accuracy
log "=== Accuracy test done ==="

log "=== Compare test start ==="
python -u run_test.py --mode compare
log "=== Compare test done ==="

log "=== All tests finished ==="
