import json,os,torch,sys,jieba
from rouge_chinese import Rouge
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
        filename = '../LLMData/按任务打包/涉法舆情摘要/train0.jsonl'
        rouge = Rouge()
        count = rouge1_f = rouge2_f = rougel_f = t_len = s_len = 0
        with (open(filename, 'r', encoding='utf-8') as file):
            for line in file:
                json_line = json.loads(line.strip())
                count += 1
                statement = json_line['text'][:1500]
                summary = json_line['summary']
                t_len += len(statement)
                s_len += len(summary)
                instruct = "请返回给定文本相应的一百字左右的摘要。"
                s = statement

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
                jieba_response = ' '.join(jieba.cut(response))
                if len(jieba_response) > 1:
                    scores = rouge.get_scores(jieba_response, ' '.join(jieba.cut(summary)))
                    # print(scores)
                    rouge1_f += scores[0]['rouge-1']['f']
                    rouge2_f += scores[0]['rouge-2']['f']
                    rougel_f += scores[0]['rouge-l']['f']
                    if count % 100 == 0:
                        print(count, summary, '=============', response)
                        print(count, float(rouge1_f / count), float(rouge2_f / count), float(rougel_f / count),
                              0.2 * float(rouge1_f / count) + 0.4 * float(rouge2_f / count) + 0.4 * float(rougel_f / count))
        print(count, float(rouge1_f / count), float(rouge2_f / count), float(rougel_f / count))




