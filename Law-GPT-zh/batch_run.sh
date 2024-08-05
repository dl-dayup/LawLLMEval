#!/bin/bash

# 定义Python脚本的路径
SCRIPT_DIR="."
# 定义要执行的Python脚本和日志文件
SCRIPT1="${SCRIPT_DIR}/Law-GPT-zh_batch_divorce.py"
LOG1="../../log/Law-GPT-zh_batch_divorce.log"
SCRIPT2="${SCRIPT_DIR}/Law-GPT-zh_batch_event.py"
LOG2="../../log/Law-GPT-zh_batch_event.log"
SCRIPT3="${SCRIPT_DIR}/Law-GPT-zh_batch_xxcq.py"
LOG3="../../log/Law-GPT-zh_batch_xxcq.log"
SCRIPT4="${SCRIPT_DIR}/Law-GPT-zh_batch_sfzy.py"
LOG4="../../log/Law-GPT-zh_batch_sfzy.log"
SCRIPT5="${SCRIPT_DIR}/Law-GPT-zh_batch_sfyqzy.py"
LOG5="../../log/Law-GPT-zh_batch_sfyqzy.log"
SCRIPT6="${SCRIPT_DIR}/Law-GPT-zh_batch_ydlj.py"
LOG6="../../log/Law-GPT-zh_batch_ydlj.log"
SCRIPT7="${SCRIPT_DIR}/Law-GPT-zh_batch_similarMatch.py"
LOG7="../../log/Law-GPT-zh_batch_similarMatch.log"
SCRIPT8="${SCRIPT_DIR}/Law-GPT-zh_batch_explainMatch.py"
LOG8="../../log/Law-GPT-zh_batch_explainMatch.log"

# 确保脚本1执行成功，然后执行脚本2，依此类推
CUDA_VISIBLE_DEVICES=1 nohup python "$SCRIPT1" > "$LOG1" 2>&1 && CUDA_VISIBLE_DEVICES=1 nohup python "$SCRIPT2" > "$LOG2" 2>&1 && CUDA_VISIBLE_DEVICES=1 nohup python "$SCRIPT3" > "$LOG3" 2>&1 && CUDA_VISIBLE_DEVICES=1 nohup python "$SCRIPT4" > "$LOG4" 2>&1 && CUDA_VISIBLE_DEVICES=1 nohup python "$SCRIPT5" > "$LOG5" 2>&1 && CUDA_VISIBLE_DEVICES=1 nohup python "$SCRIPT6" > "$LOG6" 2>&1 && CUDA_VISIBLE_DEVICES=1 nohup python "$SCRIPT7" > "$LOG7" 2>&1 && CUDA_VISIBLE_DEVICES=1 nohup python "$SCRIPT8" > "$LOG8" 2>&1

# 可选：检查最后一个脚本的执行状态
if [ $? -eq 0 ]; then
    echo "All scripts executed successfully."
else
    echo "Some scripts failed to execute."
fi