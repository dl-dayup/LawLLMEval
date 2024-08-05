import torch,json,sys
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig

fewshot_num = int(sys.argv[1]) if sys.argv[1] else 0
tokenizer = AutoTokenizer.from_pretrained("./model_baichuan2_7B_chat",use_fast=False, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("./model_baichuan2_7B_chat", device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
model.generation_config = GenerationConfig.from_pretrained("./model_baichuan2_7B_chat")
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
        example1 = '公诉机关指控，被告人黄某与吸毒人员覃某（因吸食毒品已被松滋公安局行政拘留并决定强制隔离戒毒）曾多次一起吸毒，且于2014年12月被松滋市公安局查获，但此后仍不思悔改，继续吸食毒品。2015年10月，由覃某提供毒品，被告人黄某先后3次默许并参与覃某、邓某（因吸食毒品已被松滋市公安局行政拘留并决定社 区戒毒）等人在其住所吸食毒品。具体事实如下：'
        example1_a = '人名：黄某、覃某、邓某，关系：谭某贩卖（给人）给黄某、谭某贩卖（给人）给邓某'
        example2 = '武汉市硚口区人民检察院指控：2015年5月12日凌晨，被告人凌某在武汉市硚口区中山大道钻石假日酒店对面的中国农业银行附近，以人民币300元的价格将1.11克的甲基苯丙胺片剂（俗称“麻果”）和甲基苯丙胺（冰毒）贩卖给陈某，交易 完成后被民警当场抓获归案。'
        example2_a = '人名：凌某、陈某，毒品类型：甲基苯丙胺片剂、麻果、甲基苯丙胺、冰毒，毒品重量：1.11克，关系：凌某贩卖（给人）给陈某、凌某贩卖（毒品）甲基苯丙胺、凌某贩卖（毒品）甲基苯丙胺片剂'
        example3 = '武鸣县人民检察院指控，2015年1月8日20时许，被告人黄某在武鸣县城厢镇新兴路利华隆超市门口与罗某进行毒品交易时，被公安民警当场抓获，公安民警当场当场从罗某、黄某处缴获毒品可疑物各一小包。经称量和鉴定，从罗某处查获的毒品可疑物净重2.6克，从黄某处查获的毒品可疑物净重0.5克，从中均检出氯胺酮。'
        example3_a = '人名：黄某、罗某、邓某，毒品类型：氯胺酮，毒品重量：0.5克，关系：黄某贩卖（给人）给罗某'
        example4 = '重庆市渝中区人民检察院指控，被告人吉克某某于2014年3月28日17时许，在本市渝中区南纪门下回水沟82号附近将净重0.1克的海洛因贩卖给白某某，获毒资50元，交易完成后被民警捉获，毒品毒资亦被当场缴获。公诉机关认为，被告人吉克某某的行为已构成贩卖毒品罪，提请本院依照《中华人民共和国刑法》第三百四十七条第四款、第六十七条第三款之规定判处。'
        example4_a = '人名：吉克某某、白某某，毒品类型：海洛因，关系：吉克某某贩卖（给人）给罗某、吉克某某贩卖（毒品）海洛因'
        s = statement
        messages = []
        messages.append({"role": "system", "content": instruct})
        if fewshot_num >= 1:
            messages.append({"role": "user", "content": example1})
            messages.append({"role": "assistant", "content": example1_a})
        if fewshot_num >= 2:
            messages.append({"role": "user", "content": example2})
            messages.append({"role": "assistant", "content": example2_a})
        if fewshot_num >= 3:
            messages.append({"role": "user", "content": example3})
            messages.append({"role": "assistant", "content": example3_a})
        if fewshot_num >= 4:
            messages.append({"role": "user", "content": example4})
            messages.append({"role": "assistant", "content": example4_a})
        messages.append({"role": "user", "content": s})
        response = model.chat(tokenizer, messages)
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
            print(entitys, relations, "======", response)
            print(count, float(macro_f / count))

micro_p = micro_tp/(micro_tp+micro_fn)
micro_r = micro_tp/micro_label
micro_f = 2*micro_p*micro_r/(micro_p+micro_r)
print(count,micro_f,float(macro_f/count),(micro_f+float(macro_f/count))/2)