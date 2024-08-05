import json

input_file = '../LLMData/按任务打包/可解释类案匹配/competition_stage_2_train0.json'
output_file = '../LLMData/按任务打包/可解释类案匹配/mytrain0.json'
label_dict= {0:"完全不相关", 1:"基本不相关", 2:"比较相关", 3:"非常相关"}
all_labels=label_dict.values()
count=num=right=dlen=0

with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
    for line in infile:
        json_line = json.loads(line.strip())
        count += 1
        label = json_line['label']
        text_a = json_line['Case_A']
        textA = "".join(a for a in text_a)
        text_b = json_line['Case_B']
        textB = "".join(b for b in text_b)
        data={'Case_A':textA,'Case_B':textB,'label':label}
        json_line = json.dumps(data, ensure_ascii=False)
        outfile.write(json_line + '\n')
print(count,right)