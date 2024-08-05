import torch,json

filename = '../../LLMData/按任务打包/相似案例匹配/valid.json'
count=num=right=dlen=0
with open(filename, 'r', encoding='utf-8') as file:
    for line in file:
        json_line = json.loads(line.strip())
        count += 1
        label = json_line['label']
        text_a = json_line['A']
        text_b = json_line['B']
        text_c = json_line['C']
        s = text_a + "下面B、C两个句子哪个和上述文本更相似，如果B更相似返回B否则返回C，不需要重复原文内容，只返回答案部分即可。B：" + text_b + "。C：" + text_c
        dlen+=len(s)
print(count,float(dlen/count))