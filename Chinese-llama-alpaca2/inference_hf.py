import json,os,torch,sys
from transformers import AutoModelForCausalLM, LlamaForCausalLM, LlamaTokenizer
from transformers import GenerationConfig,BitsAndBytesConfig
from peft import  PeftModel

DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant. 你是一个乐于助人的助手。"""
TEMPLATE = ("[INST] <<SYS>>\n" "{system_prompt}\n" "<</SYS>>\n\n" "{instruction} [/INST]")

model_path= "../chinese-alpaca-2-7b/"
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

generation_config = GenerationConfig(
    temperature=0.2,
    top_k=40,
    top_p=0.9,
    do_sample=True,
    num_beams=1,
    repetition_penalty=1.1,
    max_new_tokens=400
)
def generate_prompt(instruction, system_prompt=DEFAULT_SYSTEM_PROMPT):
    return TEMPLATE.format_map({'instruction': instruction,'system_prompt': system_prompt})

if __name__ == '__main__':
    load_type = torch.float16
    if torch.cuda.is_available():
        device = torch.device(0)
    else:
        device = torch.device('cpu')
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
    examples = ["请以json格式列出上述句子中涉及到的人名、地名、时间、毒品类型、毒品重量，同时判断这些人物和毒品的关系是[贩卖（给人）、贩卖（毒品）、持有、非法容留]中的哪一种，给出关系的时候需要指明头尾实体是具体的人物或毒品类型.经审理查明，2015年7月22日1时许，公安民警接到群众吴某某举报称贵阳市云岩区纯味园宿舍有一男子持有大量毒品。公安民警接警后前往举报地点搜查。在搜查过程中，从被告人焦某某身上查获毒品一包，经刑事科学技术鉴定检出海洛因计重120克。涉案毒品已上交省公安厅禁毒总队。",
                "请回答如下单项选择题，只需返回一个正确选项。1804年的《法国民法典》是世界近代法制史上的第一部民法典，是大陆法系的核心和基础。下列关于《法国民法典》的哪一项表述不正确?A:该法典体现了“个人最大限度的自由，法律最小限度的干涉”这一立法精神, B:该法典具有鲜明的革命性和时代性, C:该法典的影响后来传播到美洲、非洲和亚洲广大地区, D:该法典首次全面规定了法人制度"]
    with torch.no_grad():
        for index, example in enumerate(examples):
            input_text = generate_prompt(instruction=example, system_prompt=DEFAULT_SYSTEM_PROMPT)
            inputs = tokenizer(input_text,return_tensors="pt")  #add_special_tokens=False ?
            generation_output = model.generate(
                input_ids = inputs["input_ids"].to(device),
                attention_mask = inputs['attention_mask'].to(device),
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                generation_config = generation_config
            )
            output = tokenizer.decode(generation_output[0],skip_special_tokens=True)
            response = output.split("[/INST]")[1].strip()
            print(f"======={index}=======")
            print(f"Input: {example}\n")
            print(f"Output: {response}\n")
            # results.append({"Input":input_text,"Output":response})