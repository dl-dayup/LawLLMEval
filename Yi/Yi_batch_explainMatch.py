import torch,json
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = './model_yi_6b_chat/'
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype='auto'
).eval()
filename = '../LLMData/按任务打包/可解释类案匹配/competition_stage_2_train0.json'
label_dict= {0:"完全不相关", 1:"基本不相关", 2:"比较相关", 3:"非常相关"}
all_labels=label_dict.values()
count=num=right=dlen=0
with open(filename, 'r', encoding='utf-8') as file:
    for line in file:
        json_line = json.loads(line.strip())
        count += 1
        label = json_line['label']
        text_a = json_line['Case_A']
        textA = "".join(a for a in text_a)
        text_b = json_line['Case_B']
        textB = "".join(b for b in text_b)
        relation = json_line['relation']
        matchs=[]
        for r in relation:
            print((text_a[r[0]],text_b[r[1]]))
            matchs.append((text_a[r[0]],text_b[r[1]]))

        example = '参考如下例子回答单项选择题。例子1：以下有关民事简易程序的说法中，正确的有?A：简易程序的审限为三个月，属于法定不可变期间。B：法院在审理案件过程中发现案情复杂，需要转为普通程序的，应当在审限届满前作出决定并通知当事人。C：简易程序的开庭方式较为灵活便捷，经过双方当事人同意，法院可以采用同步视频等视听传输技术的方式开庭。D：适用简易程序案件的举证期限可以由法院确定，也可由当事人协商并经法院准许，但不得超过30日.答案：C。'
        example2 = '例子2：下列关于自然人民事权利能力和民事行为能力法律适用的说法，符合我国《涉外民事关系法律适用法》规定的是:A：自然人的民事权利能力，适用经 常居所地法律，但涉及婚姻家庭、继承的除外。B：自然人从事民事活动，依照经常居所地法律为无民事行为能力，依照行为地法律为有民事行为能力的，一律适用行为地法律。C：自然人的民事权利能力，一律适用经常居所地法律。D：自然人的民事行为能力，一律适用经常居所地法律。答案：C。'
        s = "根据A、B两段话的匹配程度，找出两段话中匹配的句子，并判断属于以下标签中的哪一个？[非常相关、比较相关、基本不相关、完全不相关] A：" + textA + "。B：" + textB
        dlen+=len(s)
        # print('statement', s)
        messages = [{"role": "user", "content": s[:4090]}]
        input_ids = tokenizer.apply_chat_template(conversation=messages, tokenize=True, add_generation_prompt=True,
                                                  return_tensors='pt')
        output_ids = model.generate(input_ids.to('cuda'))
        response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
        # print(answer, response)
        if label_dict[label] in response:
            right += 1
        if '完全不相关' in response and '基本不相关' in response and '比较相关' in response and '非常相关' in response and '答案：' + label_dict[label] not in response:
            right -= 1
        print(count)
print(count,right,float(right/count),float(dlen/count))