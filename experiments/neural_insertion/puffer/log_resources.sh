#!/usr/bin/env bash
# Sample GPU and host resources every INTERVAL seconds while a PufferLib trainer
# runs, then stop. Usage: log_resources.sh OUT.csv [INTERVAL]
set -euo pipefail
OUT=$1
INTERVAL=${2:-5}
echo "utc,gpu_util_pct,gpu_mem_util_pct,gpu_mem_used_mib,gpu_mem_total_mib,gpu_power_w,gpu_temp_c,gpu_sm_clock_mhz,load1,cpu_cores,trainer_cpu_pct,trainer_rss_mib" > "$OUT"
CORES=$(nproc)
# Wait up to a minute for the trainer to start, then sample until it exits.
for _ in $(seq 60); do pgrep -x puffer >/dev/null && break; sleep 1; done
while pgrep -x puffer >/dev/null; do
    GPU=$(nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu,clocks.sm \
          --format=csv,noheader,nounits | head -1 | tr -d ' ')
    LOAD=$(cut -d' ' -f1 /proc/loadavg)
    PROC=$(ps -C puffer -o %cpu=,rss= | awk '{c+=$1; r+=$2} END {printf "%.0f,%.0f", c, r/1024}')
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ),$GPU,$LOAD,$CORES,$PROC" >> "$OUT"
    sleep "$INTERVAL"
done
