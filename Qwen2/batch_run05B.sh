#!/bin/bash

# 定义Python脚本的路径
SCRIPT_DIR="."
MYARGS="./Qwen2-0.5B/"
# 定义要执行的Python脚本和日志文件
SCRIPT0="${SCRIPT_DIR}/Qwen2_batch_sfks_s.py"
LOG0="../log/Qwen2-0.5B_batch_sfks_s.log"
SCRIPT1="${SCRIPT_DIR}/Qwen2_batch_divorce.py"
LOG1="../log/Qwen2-0.5B_batch_divorce.log"
SCRIPT2="${SCRIPT_DIR}/Qwen2_batch_event.py"
LOG2="../log/Qwen2-0.5B_batch_event.log"
SCRIPT3="${SCRIPT_DIR}/Qwen2_batch_xxcq.py"
LOG3="../log/Qwen2-0.5B_batch_xxcq.log"
SCRIPT4="${SCRIPT_DIR}/Qwen2_batch_sfzy.py"
LOG4="../log/Qwen2-0.5B_batch_sfzy.log"
SCRIPT5="${SCRIPT_DIR}/Qwen2_batch_sfyqzy.py"
LOG5="../log/Qwen2-0.5B_batch_sfyqzy.log"
SCRIPT6="${SCRIPT_DIR}/Qwen2_batch_ydlj.py"
LOG6="../log/Qwen2-0.5B_batch_ydlj.log"
SCRIPT7="${SCRIPT_DIR}/Qwen2_batch_argument.py"
LOG7="../log/Qwen2-0.5B_batch_argument.log"
SCRIPT8="${SCRIPT_DIR}/Qwen2_batch_explainMatch.py"
LOG8="../log/Qwen2-0.5B_batch_explainMatch.log"

# 确保脚本1执行成功，然后执行脚本2，依此类推
CUDA_VISIBLE_DEVICES=1 nohup python "$SCRIPT0"  "$MYARGS" > "$LOG0" 2>&1 && CUDA_VISIBLE_DEVICES=1 nohup python "$SCRIPT1"  "$MYARGS" > "$LOG1" 2>&1 && CUDA_VISIBLE_DEVICES=1 nohup python "$SCRIPT2"  "$MYARGS" > "$LOG2" 2>&1 && CUDA_VISIBLE_DEVICES=1 nohup python "$SCRIPT3"  "$MYARGS" > "$LOG3" 2>&1 && CUDA_VISIBLE_DEVICES=1 nohup python "$SCRIPT4"  "$MYARGS" > "$LOG4" 2>&1 && CUDA_VISIBLE_DEVICES=1 nohup python "$SCRIPT5"  "$MYARGS" > "$LOG5" 2>&1 && CUDA_VISIBLE_DEVICES=1 nohup python "$SCRIPT6"  "$MYARGS" > "$LOG6" 2>&1 && CUDA_VISIBLE_DEVICES=1 nohup python "$SCRIPT7"  "$MYARGS" > "$LOG7" 2>&1 && CUDA_VISIBLE_DEVICES=1 nohup python "$SCRIPT8"  "$MYARGS" > "$LOG8" 2>&1

# 可选：检查最后一个脚本的执行状态
if [ $? -eq 0 ]; then
    echo "All scripts executed successfully."
else
    echo "Some scripts failed to execute."
fi