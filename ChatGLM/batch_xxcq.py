import os,json
import platform
from typing import Optional, Union
from transformers import AutoModel, AutoTokenizer, LogitsProcessorList
from evaluate import evaluate

MODEL_PATH = "../models/ChatGLM3-6b/"
TOKENIZER_PATH = "../models/ChatGLM3-6b/"
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True, device_map="auto").eval()
os_name = platform.system()
clear_command = "cls" if os_name == "Windows" else "clear"
stop_stream = False
welcome_prompt = "欢迎使用 ChatGLM3-6B 模型，输入内容即可进行对话，clear 清空对话历史，stop 终止程序"

def build_prompt(history):
    prompt = welcome_prompt
    for query, response in history:
        prompt += f"\n\n用户：{query}"
        prompt += f"\n\nChatGLM3-6B：{response}"
    return prompt

def process_model_outputs(outputs, tokenizer):
    responses = []
    for output in outputs:
        response = tokenizer.decode(output, skip_special_tokens=True)
        response = response.replace("[gMASK]sop", "").strip()
        batch_responses.append(response)
    return responses

def batch(
        model,
        tokenizer,
        prompts: Union[str, list[str]],
        max_length: int = 8192,
        num_beams: int = 1,
        do_sample: bool = True,
        top_p: float = 0.8,
        temperature: float = 0.8,
        logits_processor: Optional[LogitsProcessorList] = LogitsProcessorList(),
):
    tokenizer.encode_special_tokens = True
    if isinstance(prompts, str):
        prompts = [prompts]
    batched_inputs = tokenizer(prompts, return_tensors="pt", padding="longest")
    batched_inputs = batched_inputs.to(model.device)

    eos_token_id = [
        tokenizer.eos_token_id,
        tokenizer.get_command("<|user|>"),
        tokenizer.get_command("<|assistant|>"),
    ]
    gen_kwargs = {
        "max_length": max_length,
        "num_beams": num_beams,
        "do_sample": do_sample,
        "top_p": top_p,
        "temperature": temperature,
        "logits_processor": logits_processor,
        "eos_token_id": eos_token_id,
    }
    batched_outputs = model.generate(**batched_inputs, **gen_kwargs)
    batched_response = []
    for input_ids, output_ids in zip(batched_inputs.input_ids, batched_outputs):
        decoded_text = tokenizer.decode(output_ids[len(input_ids):])
        batched_response.append(decoded_text.strip())
    return batched_response

def main(batch_queries):
    gen_kwargs = {
        "max_length": 512,
        "do_sample": True,
        "top_p": 0.8,
        "temperature": 0.8,
        "num_beams": 1,
    }
    batch_responses = batch(model, tokenizer, batch_queries, **gen_kwargs)
    return batch_responses

def split_lists(list1, list2, size=40):
     if len(list1) != len(list2):
         raise ValueError("两个列表的长度必须相同")
     result = []
     num_groups = (len(list1) + size - 1) // size  # 使用整除和向上取整
     for i in range(num_groups):
        start_index = i * size
        end_index = min((i + 1) * size, len(list1))
        result.append((list1[start_index:end_index], list2[start_index:end_index]))
     return result

if __name__ == "__main__":
    batch_queries = []
    lables = []
    predicts = []
    label_dict = {'sell_drugs_to': '贩卖（给人）','traffic_in':'贩卖（毒品）','posess':'持有','provide_shelter_for':'非法容留'}
    label_list = label_dict.values()
    filename='./信息抽取/train.json'
    count = micro_tp = micro_fn = micro_label = micro_p = micro_r = micro_f = macro_f = 0
    with open(filename, 'r', encoding='utf-8') as file:
            lines=file.readlines()
            for line in lines:
              data = json.loads(line.strip())
              out_text = data['sentText']
              entityMentions = data["entityMentions"]
              entitys = set([i['text'] for i in entityMentions])
              relationMentions = data["relationMentions"]
              # relations = set([(i['em1Text'], label_dict[i['label']], i['em2Text']) for i in relationMentions])
              relations = set([label_dict[i['label']] for i in relationMentions if i['label'] != 'NA'])
              instruct = "列出下面句子中涉及到的人名、地名、时间、毒品类型、毒品重量，同时判断这些人物和毒品的关系是贩卖（给人），贩卖（毒品），持有，非法容留中的哪一种，给出关系的时候需要指名首尾两方。"
              if len(out_text)<280:
                lables.append(entitys | relations)#[dict1[i] for i in correct_answer])
                batch_queries.append( "<|user|>\n"+instruct + out_text+"\n<|assistant|>")
    split_result = split_lists(batch_queries, lables)
    for i, (bqs, bls) in enumerate(split_result):
        print("=batch" * 10,i)
        batch_responses = main(bqs)
        for response,label in zip(batch_responses,bls):
            count += 1
            tp = 0
            fn = 0
            for answer in label:
                if answer in response:
                    tp += 1
            for l in label_list:
                if l in response and l not in label:
                    fn += 1
            if tp+fn > 0:
                p=tp/(tp+fn)
            else:
                p=0
            r=tp/(len(label))
            if (p+r)==0:
                f=0
            else:
                f=2 * p * r / (p + r)
            macro_f += f
            micro_tp += tp
            micro_fn += fn
            micro_label += len(label)
            print(count)

micro_p = micro_tp/(micro_tp+micro_fn)
micro_r = micro_tp/micro_label
micro_f = 2*micro_p*micro_r/(micro_p+micro_r)
print(count,micro_f,float(macro_f/count),(micro_f+float(macro_f/count))/2)