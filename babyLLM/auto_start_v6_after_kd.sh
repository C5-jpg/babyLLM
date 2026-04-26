#!/bin/bash
# ============================================================
# 等待 V5 KD 训练完成后自动启动 V6 流水线
# 增加显存释放检查，避免 OOM
# ============================================================

KD_LOG="/home/kehe/babyllm/babyLLM/output/train_v5_phase2_kd.log"
PIPELINE_LOG="/home/kehe/babyllm/babyLLM/output/v6_pipeline.log"
LAUNCH_SCRIPT="/home/kehe/babyllm/babyLLM/launch_v6_pipeline.sh"
MONITOR_LOG="/home/kehe/babyllm/babyLLM/output/v6_auto_start_monitor.log"

wait_gpu_free() {
    local max_wait=600
    local waited=0
    echo "等待 GPU 显存释放..." | tee -a "$MONITOR_LOG"
    while [ $waited -lt $max_wait ]; do
        local total_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END {print s}')
        if [ "$total_used" -lt 10000 ]; then
            echo "GPU 显存已释放 (总占用: ${total_used}MB): $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$MONITOR_LOG"
            return 0
        fi
        echo "  GPU 总占用: ${total_used}MB, 等待中... ($waited/$max_wait 秒)" | tee -a "$MONITOR_LOG"
        sleep 15
        waited=$((waited + 15))
    done
    echo "警告: 等待超时，GPU 显存仍被占用，尝试继续启动" | tee -a "$MONITOR_LOG"
    return 1
}

echo "V5 KD -> V6 自动启动监控开始: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$MONITOR_LOG"
echo "等待 V5 KD 训练完成..." | tee -a "$MONITOR_LOG"

while true; do
    if [ -f "$KD_LOG" ]; then
        if grep -q "Phase 2.*训练完成\|Phase 2.*complete\|Phase 2 知识蒸馏.*完成" "$KD_LOG" 2>/dev/null; then
            echo "V5 KD 训练已检测完成: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$MONITOR_LOG"
            break
        fi
    fi

    PID_COUNT=$(pgrep -f "train_v5.py.*--phase kd" | wc -l 2>/dev/null || echo "0")
    if [ "$PID_COUNT" -eq 0 ]; then
        echo "V5 KD 进程已退出 (无匹配进程): $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$MONITOR_LOG"
        if [ -f "$KD_LOG" ]; then
            echo "最后 20 行日志:" >> "$MONITOR_LOG"
            tail -20 "$KD_LOG" >> "$MONITOR_LOG"
        fi
        break
    fi

    sleep 30
done

wait_gpu_free

sleep 10

echo "启动 V6 流水线: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$MONITOR_LOG"

cd /home/kehe/babyllm/babyLLM
source /home/kehe/anaconda3/etc/profile.d/conda.sh
conda activate data 2>/dev/null || true

bash "$LAUNCH_SCRIPT" 2>&1 | tee -a "$MONITOR_LOG"

echo "V6 流水线执行结束: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$MONITOR_LOG"
