import torch,json
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = './model_yi_6b_chat/'
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype='auto'
).eval()
filename = '../LLMData/按任务打包/阅读理解/ydlj_train.json'
count=num=right=dlen=0
with open(filename, 'r', encoding='utf-8') as file:
    data = json.load(file)['data']
    for d in data:
        count += 1
        p = d['paragraphs'][0]
        context = p['context']
        qas = p['qas'][0]
        casename = p['casename']
        question = qas['question']
        answers_q = qas['answers'][0]
        answers=[]
        for i in answers_q:
            answers.append(i['text'])
        example = ''
        s = example + context + '请问在这个'+casename+'中，'+question+'，如果文中没有提及请说无法回答'
        # print('statement', s)
        messages = [{"role": "user", "content": s[:4090]}]
        input_ids = tokenizer.apply_chat_template(conversation=messages, tokenize=True, add_generation_prompt=True,
                                                  return_tensors='pt')
        output_ids = model.generate(input_ids.to('cuda'))
        response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
        print(response, '=======', answers, "+++++++++" + s)
        tem=0
        for i in answers:
            if i in response:
                tem += 1
        if tem == len(answers):
            right += 1
        print(count,right,float(right/count))
print(count,right,float(right/count))