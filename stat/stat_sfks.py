import json
count=s1=s2=dlen1=dlen2=0
filename = '../LLMData/按任务打包/司法考试/0_train.json'

with open(filename, 'r', encoding='utf-8') as file:
    for line in file:
        # 去除每行末尾的换行符并解析JSON
        json_line = json.loads(line.strip())
        count+=1
        answers = json_line['answer']
        option_list = json_line['option_list']
        statement = json_line['statement']
        s = statement + ".A" + option_list['A'] + ".B" + option_list['B'] + ".C" + \
            option_list['C'] + ".D" + option_list['D']
        if len(answers) >1:
            s2+=1
            dlen2+=len(s)
        else:
            s1+=1
            dlen1+=len(s)
print(count,s1,float(dlen1/s1),s2,float(dlen2/s2))
filename = '../LLMData/按任务打包/司法考试/1_train.json'
with open(filename, 'r', encoding='utf-8') as file:
    for line in file:
        json_line = json.loads(line.strip())
        count+=1
        answers = json_line['answer']
        option_list = json_line['option_list']
        statement = json_line['statement']
        s = statement + ".A" + option_list['A'] + ".B" + option_list['B'] + ".C" + \
            option_list['C'] + ".D" + option_list['D']
        if len(answers) > 1:
            s2 += 1
            dlen2 += len(s)
        else:
            s1 += 1
            dlen1 += len(s)
print(count,s1,float(dlen1/s1),s2,float(dlen2/s2))
