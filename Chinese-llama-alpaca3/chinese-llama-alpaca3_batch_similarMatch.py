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
        filename = '../LLMData/按任务打包/相似案例匹配/valid.json'
        count = num = right = dlen = 0
        with open(filename, 'r', encoding='utf-8') as file:
            for line in file:
                json_line = json.loads(line.strip())
                count += 1
                label = json_line['label']
                text_a = json_line['A']
                text_b = json_line['B']
                text_c = json_line['C']
                example = '参考如下例子回答单项选择题。例子1：以下有关民事简易程序的说法中，正确的有?A：简易程序的审限为三个月，属于法定不可变期间。B：法院在审理案件过程中发现案情复杂，需要转为普通程序的，应当在审限届满前作出决定并通知当事人。C：简易程序的开庭方式较为灵活便捷，经过双方当事人同意，法院可以采用同步视频等视听传输技术的方式开庭。D：适用简易程序案件的举证期限可以由法院确定，也可由当事人协商并经法院准许，但不得超过30日.答案：C。'
                example2 = '例子2：下列关于自然人民事权利能力和民事行为能力法律适用的说法，符合我国《涉外民事关系法律适用法》规定的是:A：自然人的民事权利能力，适用经 常居所地法律，但涉及婚姻家庭、继承的除外。B：自然人从事民事活动，依照经常居所地法律为无民事行为能力，依照行为地法律为有民事行为能力的，一律适用行为地法律。C：自然人的民事权利能力，一律适用经常居所地法律。D：自然人的民事行为能力，一律适用经常居所地法律。答案：C。'
                instruct = "下面B、C两个句子哪个和给定文本更相似，如果B更相似返回B否则返回C，不需要重复原文内容，只返回答案部分即可。"
                s = text_a + "。B：" + text_b + "。C：" + text_c

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
                if label in response:
                    right += 1
                if 'B' in response and 'C' in response:
                    right -= 1
                if count % 100 == 0:
                    print(count, label, '=============', response)
                    print(count, right, float(right / count))
        print(count, right, float(right / count))




