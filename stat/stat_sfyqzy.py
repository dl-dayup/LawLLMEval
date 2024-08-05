import torch,json
from rouge import Rouge

filename = '../LLMData/按任务打包/涉法舆情摘要/train.jsonl'
rouge=Rouge()
count = rouge1_f = rouge2_f = rougel_f = t_len = s_len = 0
with (open(filename, 'r', encoding='utf-8') as file):
    for line in file:
        json_line = json.loads(line.strip())
        count+=1
        statement = json_line['text']
        summary = json_line['summary']
        t_len += len(statement)
        s_len += len(summary)
print(count, t_len/count, s_len/count)