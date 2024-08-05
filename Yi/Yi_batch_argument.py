import torch,json
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = './model_yi_6b_chat/'
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype='auto'
).eval()
filename = '../LLMData/按任务打包/论辩理解/stage_1/train_entry0.jsonl'
count=num=right=dlen=0
with open(filename, 'r', encoding='utf-8') as file:
    for line in file:
        json_line = json.loads(line.strip())
        count += 1
        sc = json_line['sc']
        crime = json_line['crime']
        bc_1 = json_line['bc_1']
        bc_2 = json_line['bc_2']
        bc_3 = json_line['bc_3']
        bc_4 = json_line['bc_4']
        bc_5 = json_line['bc_5']
        label = json_line['answer']
        s = sc + "在这个关于"+crime+"的描述中，判断以下五个句子中哪个句子与给定文本能组成有争议的观点对，请返回句子号。1："+bc_1+"。2："+bc_2+"。3："+bc_3+"。4: "+bc_4+"。5: "+bc_5
        dlen+=len(s)
        messages = [{"role": "user", "content": s[:4090]}]
        input_ids = tokenizer.apply_chat_template(conversation=messages, tokenize=True, add_generation_prompt=True,
                                                  return_tensors='pt')
        output_ids = model.generate(input_ids.to('cuda'))
        response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
        if str(label) in response:
            right+=1
        if '1' in response and '2' in response and '答案：' + str(label) not in response:
            right -= 1
        if count % 100 == 0:
            print(count, label, '=============', response)
            print(count, right, float(right / count))
        # if 'A' in response and 'B' in response and 'C' in response and 'D' in response and '答案：' + answer not in response:
        #     right -= 1
print(count,right,float(right/count),float(dlen/count))