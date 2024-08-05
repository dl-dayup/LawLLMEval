import torch,json

filename = '../../LLMData/按任务打包/divorce/dev.txt'
count = dlen = llen =0
with (open(filename, 'r', encoding='utf-8') as file):
    for line in file:
        json_line = json.loads(line.strip())
        count+=1
        statement = json_line['text_a']
        labels = json_line['labels']
        dlen+=len(statement)
        llen+=len(labels)

print(count,float(dlen/count),float(llen/count))