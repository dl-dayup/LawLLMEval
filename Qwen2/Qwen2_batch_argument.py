import sys,json
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = sys.argv[1] #
device = "cuda" # the device to load the model onto
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

filename = '../LLMData/按任务打包/论辩理解/stage_1/train_entry0.jsonl'
count=num=right=dlen=0
with (open(filename, 'r', encoding='utf-8') as file):
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
        instruct = "在这个关于"+crime+"的描述中，以下五个句子中有一个句子可以与给定文本能组成有争议的观点对，请判断是哪一个句子并返回句子号。不需要重复原文内容和给定的五个句子内容。"
        s = sc + "1："+bc_1+"。2："+bc_2+"。3："+bc_3+"。4: "+bc_4+"。5: "+bc_5
        messages = [
            {"role": "system", "content": instruct},
            {"role": "user", "content": s}]
        text = tokenizer.apply_chat_template(messages[:3000], tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(device)
        generated_ids = model.generate(**model_inputs, max_new_tokens=100)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in
                         zip(model_inputs.input_ids, generated_ids)]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        if str(label) in response:
            right += 1
        if '1' in response and '2' in response and '答案：' + str(label) not in response:
            right -= 1
        if count%100 == 0:
            print(count, label, '=============', response)
            print(count, right, float(right / count))

print(count,right,float(right/count),float(dlen/count))