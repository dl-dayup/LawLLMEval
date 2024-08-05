import json, os, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import GenerationConfig, BitsAndBytesConfig

DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant. 你是一个乐于助人的助手。"""
system_format='<|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>'
user_format='<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
assistant_format='{content}<|eot_id|>'
model_path= "../llama-3-chinese-8b-instruct-v3/"
generation_config = GenerationConfig(
    temperature=0.2,
    top_k=40,
    top_p=0.9,
    do_sample=True,
    num_beams=1,
    repetition_penalty=1.1,
    max_new_tokens=400
)
def generate_prompt(instruction, system_prompt):
    return system_format.format(content=system_prompt) + user_format.format(content=instruction)

if __name__ == '__main__':
    load_type = torch.float16
    device = torch.device('cuda')
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=load_type,
        low_cpu_mem_usage=True,
        device_map='auto',
        quantization_config=None,
        attn_implementation="sdpa"
    ).eval()

    with torch.no_grad():
        filename = '../LLMData/按任务打包/阅读理解/ydlj_train.json'
        count = num = right = dlen = 0
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
                answers = []
                for i in answers_q:
                    answers.append(i['text'])
                example = ''
                instruct = '请问在这个' + casename + '中，' + question + '，不需要重复原文内容，只返回答案部分即可，如果文中没有提及请返回无法回答'
                s = example + context
                input_text = generate_prompt(instruction=s, system_prompt=instruct)
                inputs = tokenizer(input_text, return_tensors="pt")  # add_special_tokens=False ?
                generation_output = model.generate(
                    input_ids=inputs["input_ids"].to(device),
                    attention_mask=inputs['attention_mask'].to(device),
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    generation_config=generation_config
                )
                output = tokenizer.decode(generation_output[0], skip_special_tokens=True)
                response = output.split("assistant\n\n")[1].strip()
                tem = 0
                for i in answers:
                    if i in response:
                        tem += 1
                if tem == len(answers):
                    right += 1
                if count % 100 == 0:
                    print(count, answers, '=============', response)
                    print(count, right, float(right / count))
        print(count, right, float(right / count))




