import sys,json
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = sys.argv[1] #
device = "cuda" # the device to load the model onto
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

filename = '../LLMData/按任务打包/阅读理解/ydlj_train.json'
count=num=right=dlen=0
with open(filename, 'r', encoding='utf-8') as file:
    data = json.load(file)['data']
    for d in data:
        count += 1
        p = d['paragraphs'][0]
        context = p['context']
        casename = p['casename']
        qas = p['qas'][0]
        question = qas['question']
        answers_q = qas['answers'][0]
        answers=[]
        for i in answers_q:
            answers.append(i['text'])
        example = ''
        instruct = '请问在这个'+casename+'中，'+question+'，不需要重复原文内容，只返回答案部分即可，如果文中没有提及请返回无法回答'
        s = example + context
        # print('statement', s)
        messages = [
            {"role": "system", "content": instruct},
            {"role": "user", "content": s}]
        text = tokenizer.apply_chat_template(messages[:3000], tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(device)
        generated_ids = model.generate(**model_inputs, max_new_tokens=100)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in
                         zip(model_inputs.input_ids, generated_ids)]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        tem=0
        for i in answers:
            if i in response:
                tem += 1
        if tem == len(answers):
            right += 1
        if count % 100 == 0:
            print(count, answers, '=============', response)
            print(count,right,float(right/count))
print(count,right,float(right/count))