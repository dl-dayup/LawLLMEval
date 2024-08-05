import torch,json,sys
from transformers import AutoModelForCausalLM, AutoTokenizer

fewshot_num = int(sys.argv[1]) if sys.argv[1] else 0
model_path = './model_yi_6b_chat/'
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype='auto'
).eval()
filename = '../LLMData/按任务打包/divorce/dev.txt'
label_dict= {0:"婚后有子女", 1:"限制行为能力子女抚养", 2:"有夫妻共同财产", 3:"支付抚养费", 4:"不动产分割", 5:"婚后分居", 6:"二次起诉离婚", 7:"按月给付抚养费", 8:"准予离婚",
            9:"有夫妻共同债务", 10:"婚前个人财产", 11:"法定离婚", 12:"不履行家庭义务", 13:"存在非婚生子", 14:"适当帮助", 15:"不履行离婚协议", 16:"损害赔偿", 17:"感情不和分居满二年",
            18:"子女随非抚养权人生活", 19:"婚后个人财产"}
all_labels=label_dict.values()
count = micro_tp = micro_predict = micro_label = micro_p = micro_r = micro_f = macro_f = 0
with (open(filename, 'r', encoding='utf-8') as file):
    for line in file:
        json_line = json.loads(line.strip())
        count+=1
        statement = json_line['text_a']
        labels = json_line['labels']
        answers = [label_dict[l] for l in labels]
        instruct = "请进行多标签分类，判断上述文本中涉及到以下标签中的哪几个标签？[婚后有子女、限制行为能力子女抚养、有夫妻共同财产、支付抚养费、不动产分割、婚后分居、二次起诉离婚、按月给付抚养费、准予离婚、有夫妻共同债务、婚前个人财产、法定离婚、不履行家庭义务、存在非婚生子、适当帮助、不履行离婚协议、损害赔偿、感情不和分居满二年、子女随非抚养权人生活、婚后个人财产]"
        example1 = '被告从2010年开始沉迷赌博，对家庭不管不顾，从2012年开始便离家出走，至今下落不明。'
        example1_a = '不履行家庭义务、感情不和分居满二年'
        example2 = '2010年3月初，原告为了与我离婚，抛下女儿，将家里的现金全部带走，与我们父女俩分居生活，弄得我身无分文，为了生活不得已将帕萨特牌轿 车卖给了朋友葛峥伟，获得车价款15万元，车辆已办理过户。'
        example2_a = '不履行家庭义务、婚后分居、婚后有子女、有夫妻共同财产'
        example3 = '2、判令被告给付女儿抚养费人民币60000元（自2012年1月至2017年1月共5年计60个月每月1000元）；'
        example3_a = '婚后有子女、支付抚养费、限制行为能力子女抚养'
        example4 = '二、婚生一女方某由女方扶养到独立生活，男方每月给抚养费叁仟元到孩子大学毕业，男方有探视权；'
        example4_a = '婚后有子女、按月给付抚养费、支付抚养费、限制行为能力子女抚养'
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
        input_ids = tokenizer.apply_chat_template(conversation=messages, tokenize=True, add_generation_prompt=True,
                                                  return_tensors='pt')
        output_ids = model.generate(input_ids.to('cuda'))
        response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
        # print(answers, response)
        tp = 0
        predict = 0
        for answer in answers:
            if answer in response:
                tp += 1
        for l in all_labels:
            if l in response:
                predict += 1
        # if tp == len(answers):
        #     right += 1
        micro_tp += tp
        micro_predict += predict
        micro_label += len(answers)
        if predict>0:
            p=tp/predict
        else:
            p=0
        r=tp/len(answers)
        if (p+r)==0:
            f=0
        else:
            f=2 * p * r / (p + r)

        macro_f += f
        print(count)

micro_p = micro_tp/micro_predict
micro_r = micro_tp/micro_label
micro_f = 2*micro_p*micro_r/(micro_p+micro_r)
print(count,micro_f,float(macro_f/count),(micro_f+float(macro_f/count))/2)