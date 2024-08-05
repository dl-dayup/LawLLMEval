import json,os,torch,sys,jieba
from rouge_chinese import Rouge
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
        filename = '../LLMData/按任务打包/司法摘要/data_second_train0.json'
        rouge = Rouge()
        count = rouge1_f = rouge2_f = rougel_f = t_len = s_len = 0
        with (open(filename, 'r', encoding='utf-8') as file):
            for line in file:
                json_line = json.loads(line.strip())
                count += 1
                statement = json_line['question']
                candidates = json_line['candidates']
                candi = "。".join(ca for ca in candidates)
                summary = json_line['answer']
                t_len += len(statement) + len(candi)
                s_len += len(summary)
                s = statement + candi
                instruct = "给定问题和参考文本，你需要给出问题的六十字左右的回答。"

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
                scores = rouge.get_scores(' '.join(jieba.cut(response)), ' '.join(jieba.cut(summary)))
                # print(scores)
                rouge1_f += scores[0]['rouge-1']['f']
                rouge2_f += scores[0]['rouge-2']['f']
                rougel_f += scores[0]['rouge-l']['f']
                if count % 100 == 0:
                    print(count, summary, '=============', response)
                    print(count, float(rouge1_f / count), float(rouge2_f / count), float(rougel_f / count),
                          0.2 * float(rouge1_f / count) + 0.4 * float(rouge2_f / count) + 0.4 * float(rougel_f / count))
        print(count, float(rouge1_f / count), float(rouge2_f / count), float(rougel_f / count),
                  0.2 * float(rouge1_f / count) + 0.4 * float(rouge2_f / count) + 0.4 * float(rougel_f / count))



