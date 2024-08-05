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
        filename = '../LLMData/按任务打包/divorce/dev.txt'
        label_dict = {0: "婚后有子女", 1: "限制行为能力子女抚养", 2: "有夫妻共同财产", 3: "支付抚养费", 4: "不动产分割",
                      5: "婚后分居", 6: "二次起诉离婚", 7: "按月给付抚养费", 8: "准予离婚",
                      9: "有夫妻共同债务", 10: "婚前个人财产", 11: "法定离婚", 12: "不履行家庭义务", 13: "存在非婚生子",
                      14: "适当帮助", 15: "不履行离婚协议", 16: "损害赔偿", 17: "感情不和分居满二年",
                      18: "子女随非抚养权人生活", 19: "婚后个人财产"}
        all_labels = label_dict.values()
        count = micro_tp = micro_predict = micro_label = micro_p = micro_r = micro_f = macro_f = 0
        with (open(filename, 'r', encoding='utf-8') as file):
            for line in file:
                json_line = json.loads(line.strip())
                count += 1
                statement = json_line['text_a']
                labels = json_line['labels']
                answers = [label_dict[l] for l in labels]
                s = statement
                instruct = "请进行多标签分类，判断所述文本中涉及到以下标签中的哪几个标签？[婚后有子女、限制行为能力子女抚养、有夫妻共同财产、支付抚养费、不动产分割、婚后分居、二次起诉离婚、按月给付抚养费、准予离婚、有夫妻共同债务、婚前个人财产、法定离婚、不履行家庭义务、存在非婚生子、适当帮助、不履行离婚协议、损害赔偿、感情不和分居满二年、子女随非抚养权人生活、婚后个人财产]"

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
                tp = 0
                predict = 0
                for answer in answers:
                    if answer in response:
                        tp += 1
                for l in all_labels:
                    if l in response:
                        predict += 1
                # if tp == len(answers):
                #     right += 1
                micro_tp += tp
                micro_predict += predict
                micro_label += len(answers)
                if predict > 0:
                    p = tp / predict
                else:
                    p = 0
                r = tp / len(answers)
                if (p + r) == 0:
                    f = 0
                else:
                    f = 2 * p * r / (p + r)

                macro_f += f
                if count % 100 == 0:
                    print(count, answers, '=============', response)
                    print(count, 'marco_f:', float(macro_f / count))

        micro_p = micro_tp / micro_predict
        micro_r = micro_tp / micro_label
        micro_f = 2 * micro_p * micro_r / (micro_p + micro_r)
        print(count, micro_f, float(macro_f / count), (micro_f + float(macro_f / count)) / 2)




