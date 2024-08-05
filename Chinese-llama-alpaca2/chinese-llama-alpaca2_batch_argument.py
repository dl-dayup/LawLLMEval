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
        filename = '../LLMData/按任务打包/论辩理解/stage_1/train_entry0.jsonl'
        count = num = right = dlen = 0
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
                instruct = "在这个关于" + crime + "的描述中，以下五个句子中有一个句子可以与给定文本能组成有争议的观点对，请判断是哪一个句子并返回句子号。不需要重复原文内容和给定的五个句子内容。"
                s = sc + "1：" + bc_1 + "。2：" + bc_2 + "。3：" + bc_3 + "。4: " + bc_4 + "。5: " + bc_5
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
                if str(label) in response:
                    right += 1
                if '1' in response and '2' in response and '答案：' + str(label) not in response:
                    right -= 1
                if count % 100 == 0:
                    print(count, label, '=============', response)
                    print(count, right, float(right / count))
        print(count, right, float(right / count), float(dlen / count))




