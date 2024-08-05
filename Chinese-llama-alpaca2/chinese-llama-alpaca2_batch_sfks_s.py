import json,os,torch,sys
from transformers import AutoModelForCausalLM, LlamaForCausalLM, LlamaTokenizer
from transformers import GenerationConfig,BitsAndBytesConfig
from peft import  PeftModel

model_path= "../chinese-alpaca-2-7b/"
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
    return ("[INST] <<SYS>>\n" "{system_prompt}\n" "<</SYS>>\n\n" "{instruction} [/INST]").format_map({'instruction': instruction,'system_prompt': system_prompt})

if __name__ == '__main__':
    load_type = torch.float16
    device = torch.device('cuda')
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=load_type,
        low_cpu_mem_usage=True,
        device_map='auto',
        load_in_4bit=None,
        load_in_8bit=None,
        quantization_config=None,
        attn_implementation="sdpa",
        trust_remote_code=True
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
                    response = output.split("[/INST]")[1].strip()

                    if answer in response:
                        right += 1
                    if 'A' in response and 'B' in response and 'C' in response and 'D' in response and '答案：' + answer not in response:
                        right -= 1
                    if count % 100 == 0:
                        print(count, answer, '=============', response)
                        print(count, right, float(right / count))
        print(count, right, float(right / count))



