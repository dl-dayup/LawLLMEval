import json
count=p1=q1=im1=dlen=0
filename = '../LLMData/按任务打包/阅读理解/ydlj_train.json'

with open(filename, 'r', encoding='utf-8') as file:
        data = json.load(file)['data']
        for d in data:
            count+=1
            paragraphs=d['paragraphs']
            for p in paragraphs:
                p1+=1
                context = p['context']
                dlen+=len(context)
                qas = p['qas']
                for q in qas:
                    q1+=1
                    if q['is_impossible']=='true':
                        im1+=1
print(count,p1,q1,im1,float(dlen/p1))
