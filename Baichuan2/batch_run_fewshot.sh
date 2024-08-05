#!/bin/bash
# arg1=fewshot num, arg2=cuda device
# 定义Python脚本的路径
SCRIPT_DIR="."
# 定义要执行的Python脚本和日志文件
FSARGS=$1
SCRIPT0="${SCRIPT_DIR}/Baichuan2_batch_sfks_s.py"
LOG0="../log/Baichuan2_batch_sfks_s_fs$1.log"
SCRIPT1="${SCRIPT_DIR}/Baichuan2_batch_divorce.py"
LOG1="../log/Baichuan2_batch_divorce_fs$1.log"
SCRIPT2="${SCRIPT_DIR}/Baichuan2_batch_event.py"
LOG2="../log/Baichuan2_batch_event_fs$1.log"
SCRIPT3="${SCRIPT_DIR}/Baichuan2_batch_sfzy.py"
LOG3="../log/Baichuan2_batch_sfzy_fs$1.log"
SCRIPT4="${SCRIPT_DIR}/Baichuan2_batch_sfks_m.py"
LOG4="../log/Baichuan2_batch_sfks_m_fs$1.log"
SCRIPT5="${SCRIPT_DIR}/Baichuan2_batch_similarMatch.py"
LOG5="../log/Baichuan2_batch_similarMatch_fs$1.log"

# 确保脚本1执行成功，然后执行脚本2，依此类推
CUDA_VISIBLE_DEVICES=$2 nohup python "$SCRIPT0" "$FSARGS" > "$LOG0" 2>&1 && CUDA_VISIBLE_DEVICES=$2 nohup python "$SCRIPT1" "$FSARGS" > "$LOG1" 2>&1 && CUDA_VISIBLE_DEVICES=$2 nohup python "$SCRIPT2" "$FSARGS" > "$LOG2" 2>&1 && CUDA_VISIBLE_DEVICES=$2 nohup python "$SCRIPT3" "$FSARGS" > "$LOG3" 2>&1 && CUDA_VISIBLE_DEVICES=$2 nohup python "$SCRIPT4" "$FSARGS" > "$LOG4" 2>&1 && CUDA_VISIBLE_DEVICES=$2 nohup python "$SCRIPT5" "$FSARGS" > "$LOG5" 2>&1
# 可选：检查最后一个脚本的执行状态
if [ $? -eq 0 ]; then
    echo "All scripts executed successfully."
else
    echo "Some scripts failed to execute."
fi