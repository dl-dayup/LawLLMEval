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
        filename = '../LLMData/按任务打包/事件检测/train0.jsonl'
        count = micro_tp = micro_fn = micro_label = micro_p = micro_r = micro_f = macro_f = 0
        with (open(filename, 'r', encoding='utf-8') as file):
            for line in file:
                count += 1
                json_line = json.loads(line.strip())
                events = json_line['events']
                content = "".join(i['sentence'] for i in json_line['content'])

                negative_triggers = json_line['negative_triggers']
                answers = set([e['mention'][0]['trigger_word'] for e in events])
                negative_labels = set([n['trigger_word'] for n in negative_triggers])
                s = content
                instruct = "找出这段话中的事件类型及其触发词。"

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
                tp = 0
                fn = 0
                for answer in answers:
                    if answer in response:
                        tp += 1
                for l in negative_labels:
                    if l in response:
                        fn += 1
                if tp + fn > 0:
                    p = tp / (tp + fn)
                else:
                    p = 0
                r = tp / len(answers)
                if (p + r) == 0:
                    f = 0
                else:
                    f = 2 * p * r / (p + r)

                macro_f += f
                micro_tp += tp
                micro_fn += fn
                micro_label += len(answers)
                if count % 100 == 0:
                    print(count, answers, '=============', response)
                    print(count, 'marco_f:', float(macro_f / count))

        micro_p = micro_tp / (micro_tp + micro_fn)
        micro_r = micro_tp / micro_label
        micro_f = 2 * micro_p * micro_r / (micro_p + micro_r)
        print(count, micro_f, float(macro_f / count), (micro_f + float(macro_f / count)) / 2)




