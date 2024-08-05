import torch,json
from transformers import AutoTokenizer, AutoModel
from model.modeling_chatglm import ChatGLMForConditionalGeneration

tokenizer = AutoTokenizer.from_pretrained("model", trust_remote_code=True)
model_class = ChatGLMForConditionalGeneration
model = model_class.from_pretrained("model", device_map = "auto").half().eval()

filename = '../../LLMData/按任务打包/信息抽取/train.json'
label_dict = {"posess":"持有", "sell_drugs_to":"贩卖（给人）", "traffic_in":"贩卖（毒品）", "provide_shelter_for":"非法容留"}
label_list = label_dict.values()
count = micro_tp = micro_fn = micro_label = micro_p = micro_r = micro_f = macro_f = 0
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

        s = statement + "请列出上述句子中涉及到的人名、地名、时间、毒品类型、毒品重量，同时判断这些人物和毒品的关系是[贩卖（给人）、贩卖（毒品）、持有、非法容留]中的哪一种，给出关系的时候需要指明头尾实体是具体的人物或毒品类型"
        response = model.chat(tokenizer, s[:4096], history='', max_length=4096)
        # print(answers, response)
        tp = 0
        fn = 0
        for answer in entitys:
            if answer in response:
                tp += 1
        for answer in relations:
            if answer in response:
                tp += 1
        for l in label_list:
            if l in response and l not in relations:
                fn += 1
        if tp+fn > 0:
            p=tp/(tp+fn)
        else:
            p=0
        r=tp/(len(entitys)+len(relations))
        if (p+r)==0:
            f=0
        else:
            f=2 * p * r / (p + r)

        macro_f += f
        micro_tp += tp
        micro_fn += fn
        micro_label += len(entitys)+len(relations)
        print(count,'marco_f1:',float(macro_f/count))

micro_p = micro_tp/(micro_tp+micro_fn)
micro_r = micro_tp/micro_label
micro_f = 2*micro_p*micro_r/(micro_p+micro_r)
print(count,micro_f,float(macro_f/count),(micro_f+float(macro_f/count))/2)