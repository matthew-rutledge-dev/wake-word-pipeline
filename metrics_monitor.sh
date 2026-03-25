#!/bin/bash
# Wake Word Training Metrics Monitor
# Logs CPU, RAM, GPU, VRAM every 10 seconds to CSV
LOG_DIR="/opt/ai/wakeword-train/metrics"
mkdir -p "$LOG_DIR"
CSV="$LOG_DIR/training_metrics_$(date +%Y%m%d_%H%M%S).csv"
echo "timestamp,cpu_usr_pct,cpu_sys_pct,cpu_idle_pct,ram_used_mb,ram_total_mb,ram_pct,swap_used_mb,gpu_util_pct,gpu_mem_used_mb,gpu_mem_total_mb,gpu_temp_c,gpu_power_w,active_word,process_rss_mb" > "$CSV"
echo "Logging to: $CSV"
echo "Press Ctrl+C to stop"
while true; do
    TS=$(date +"%Y-%m-%d %H:%M:%S")
    CPU_LINE=$(mpstat 1 1 | tail -1)
    CPU_USR=$(echo "$CPU_LINE" | awk '{print $3}')
    CPU_SYS=$(echo "$CPU_LINE" | awk '{print $5}')
    CPU_IDLE=$(echo "$CPU_LINE" | awk '{print $12}')
    RAM_LINE=$(free -m | grep Mem:)
    RAM_USED=$(echo "$RAM_LINE" | awk '{print $3}')
    RAM_TOTAL=$(echo "$RAM_LINE" | awk '{print $2}')
    RAM_PCT=$(awk "BEGIN{printf \"%.1f\", ($RAM_USED/$RAM_TOTAL)*100}")
    SWAP_LINE=$(free -m | grep Swap:)
    SWAP_USED=$(echo "$SWAP_LINE" | awk '{print $3}')
    GPU_DATA=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits 2>/dev/null)
    GPU_UTIL=$(echo "$GPU_DATA" | awk -F", " '{print $1}')
    GPU_MEM_USED=$(echo "$GPU_DATA" | awk -F", " '{print $2}')
    GPU_MEM_TOTAL=$(echo "$GPU_DATA" | awk -F", " '{print $3}')
    GPU_TEMP=$(echo "$GPU_DATA" | awk -F", " '{print $4}')
    GPU_POWER=$(echo "$GPU_DATA" | awk -F", " '{print $5}')
    ACTIVE_WORD=$(grep -oP '(?<=\] )\S+' /opt/ai/wakeword-train/batch_train.log 2>/dev/null | tail -1)
    TRAIN_PID=$(pgrep -f "01_generate_samples|02_train_oww" | head -1)
    if [ -n "$TRAIN_PID" ]; then
        PROC_RSS=$(ps -o rss= -p "$TRAIN_PID" 2>/dev/null | awk '{printf "%.0f", $1/1024}')
    else
        PROC_RSS=0
    fi
    echo "$TS,$CPU_USR,$CPU_SYS,$CPU_IDLE,$RAM_USED,$RAM_TOTAL,$RAM_PCT,$SWAP_USED,$GPU_UTIL,$GPU_MEM_USED,$GPU_MEM_TOTAL,$GPU_TEMP,$GPU_POWER,$ACTIVE_WORD,$PROC_RSS" >> "$CSV"
    printf "\r[%s] CPU:%s%% RAM:%s/%sMB(%.0f%%) SWAP:%sMB GPU:%s%% VRAM:%s/%sMB %sC %sW | %s RSS:%sMB   " "$TS" "$CPU_USR" "$RAM_USED" "$RAM_TOTAL" "$RAM_PCT" "$SWAP_USED" "$GPU_UTIL" "$GPU_MEM_USED" "$GPU_MEM_TOTAL" "$GPU_TEMP" "$GPU_POWER" "$ACTIVE_WORD" "$PROC_RSS"
    sleep 9
done