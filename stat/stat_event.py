import json
count=p1=q1=im1=dlen=0
filename = '../LLMData/按任务打包/事件检测/train.jsonl'

with open(filename, 'r', encoding='utf-8') as file:
    for line in file:
        count += 1
        json_line = json.loads(line.strip())
        events = json_line['events']
        content = "".join( i['sentence'] for i in json_line['content'])

        negatvie_triggers = json_line['negative_triggers']
        labels=set([e['mention'][0]['trigger_word']  for e in events])
        negative_labels = set([n['trigger_word'] for n in negatvie_triggers])
        if count>1600 and count<1610:
            print('content:',json_line['title'],json_line['crime'],content)
            print('events:',labels)
            print('negative_triggers:',negative_labels)

#         dlen += len(content)
#         p1+=len(labels)
#         q1+=len(negative_labels)
# print(count,p1/count,q1/count,float(dlen/count))
