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
        filename = '../LLMData/按任务打包/司法考试/0_train0.json'
        count = num = right  = 0
        with open(filename, 'r', encoding='utf-8') as file:
            for line in file:
                # 去除每行末尾的换行符并解析JSON
                json_line = json.loads(line.strip())
                answers = json_line['answer']
                if len(answers) == 1:
                    count += 1
                    answer = answers[0]
                    option_list = json_line['option_list']
                    statement = json_line['statement']
                    instruct = "请回答如下单项选择题，只需返回一个正确选项。"
                    s = statement + "。A：" + option_list['A'] + "。B：" + option_list['B'] + "。C：" + option_list[
                        'C'] + "。D：" + option_list['D']
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
                    if answer in response:
                        right += 1
                    if 'A' in response and 'B' in response and 'C' in response and 'D' in response and '答案：' + answer not in response:
                        right -= 1
                    if count % 100 == 0:
                        print(count, answer, '=============', response)
                        print(count, right, float(right / count))
        print(count, right, float(right / count))



