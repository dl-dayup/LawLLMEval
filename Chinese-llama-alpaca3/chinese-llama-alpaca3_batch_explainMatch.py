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
        filename = '../LLMData/按任务打包/可解释类案匹配/mytrain0.json'  # competition_stage_2_train0.json'
        label_dict = {0: "完全不相关", 1: "基本不相关", 2: "比较相关", 3: "非常相关"}
        all_labels = label_dict.values()
        count = num = right = dlen = 0
        with open(filename, 'r', encoding='utf-8') as file:
            for line in file:
                json_line = json.loads(line.strip())
                count += 1
                label = json_line['label']
                textA = json_line['Case_A']
                textB = json_line['Case_B']
                instruct = "根据A、B两段话的匹配程度和标签的定义，进行分类任务，判断属于以下标签中的哪一个？[非常相关、比较相关、基本不相关、完全不相关]。非常相关：要件事实相关，且案情事实相关。比较相关：要件事实相关，但案情事实不相关。基本不相关：要件事实不相关，但案情事实相关。完全不相关：要件事实不相关，且案情事实不相关。"
                s = "A：" + textA + "。B：" + textB
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
                if label_dict[label] in response:
                    right += 1
                if '完全不相关' in response and '基本不相关' in response and '比较相关' in response and '非常相关' in response and '答案：' + \
                        label_dict[label] not in response:
                    right -= 1
                if count % 100 == 0:
                    print(count, label_dict[label], '=============', response)
                    print(count, right, float(right / count))
        print(count, right, float(right / count), float(dlen / count))




