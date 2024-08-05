import sys,json
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = sys.argv[1] #
device = "cuda" # the device to load the model onto
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

filename = '../LLMData/按任务打包/信息抽取/train.json'
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
        instruct = "请列出上述句子中涉及到的人名、地名、时间、毒品类型、毒品重量，同时判断这些人物和毒品的关系是[贩卖（给人）、贩卖（毒品）、持有、非法容留]中的哪一种，给出关系的时候需要指明头尾实体是具体的人物或毒品类型"
        s = statement
        messages = [
            {"role": "system", "content": instruct},
            {"role": "user", "content": s}]
        text = tokenizer.apply_chat_template(messages[:512], tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(device)
        generated_ids = model.generate(**model_inputs, max_new_tokens=200)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in
                         zip(model_inputs.input_ids, generated_ids)]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
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
        if count % 100 == 0:
            print(count, entitys, relations, '=============', response)
            print(count, 'marco_f:', float(macro_f / count))

micro_p = micro_tp/(micro_tp+micro_fn)
micro_r = micro_tp/micro_label
micro_f = 2*micro_p*micro_r/(micro_p+micro_r)
print(count,micro_f,float(macro_f/count),(micro_f+float(macro_f/count))/2)