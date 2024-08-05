import torch,json
from rouge import Rouge

filename = '../LLMData/按任务打包/司法摘要/data_second_train.json'
rouge=Rouge()
count = rouge1_f = rouge2_f = rougel_f = t_len = s_len = 0
with (open(filename, 'r', encoding='utf-8') as file):
    for line in file:
        json_line = json.loads(line.strip())
        count+=1
        statement = json_line['question']
        candidates = json_line['candidates']
        candi = "。".join(ca for ca in candidates)
        summary = json_line['answer']
        t_len += len(statement) + len(candi)
        s_len += len(summary)
print(count, t_len/count, s_len/count)