import json

filename = '../LLMData/按任务打包/信息抽取/train.json'
label_dict = {"posess":"持有", "sell_drugs_to":"贩卖（给人）", "traffic_in":"贩卖（毒品）", "provide_shelter_for":"非法容留"}
label_list = label_dict.values()
count = dlen = elen = rlen =0
with (open(filename, 'r', encoding='utf-8') as file):
    for line in file:
        count += 1
        json_line = json.loads(line.strip())
        statement = json_line['sentText']
        entityMentions = json_line["entityMentions"]
        entitys = set([ i['text'] for i in entityMentions])
        relationMentions = json_line["relationMentions"]
        # relations = set([(i['em1Text'], label_dict[i['label']], i['em2Text']) for i in relationMentions])
        relations = set([label_dict[i['label']] for i in relationMentions if i['label']!= 'NA' ])
        dlen+=len(statement)
        elen+=len(entitys)
        rlen+=len(relations)
print(count,dlen/count,elen/count,rlen/count)