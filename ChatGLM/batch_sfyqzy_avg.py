import os,json,jieba
from rouge_chinese import Rouge
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
        "max_length": 8192,
        "do_sample": True,
        "top_p": 0.8,
        "temperature": 0.8,
        "num_beams": 1,
    }
    batch_responses = batch(model, tokenizer, batch_queries, **gen_kwargs)
    return batch_responses

def split_lists(list1, list2, size=4):
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
    rouge=Rouge()
    filename = '../../LLMData/按任务打包/涉法舆情摘要/train0.jsonl'
    count = rouge1_f = rouge2_f = rougel_f = t_len = s_len = 0
    with open(filename, 'r', encoding='utf-8') as file:
            lines=file.readlines()
            for line in lines:
                json_line = json.loads(line.strip())
                count += 1
                statement = json_line['text']
                summary = json_line['summary']
                if len(statement)<3000:
                    lables.append(summary)
                    batch_queries.append( "<|user|>\n"+statement+"。请返回上述文本相应的一百字左右的摘要。\n<|assistant|>")
    split_result = split_lists(batch_queries, lables)
    r1=r2=rl=0.0
    num=0.0
    for i, (bqs, bls) in enumerate(split_result):
        print("=batch" * 10,i)
        batch_responses = main(bqs)
        for re,la in zip(batch_responses,bls):
            scores = rouge.get_scores(' '.join(jieba.cut(re)), ' '.join(jieba.cut(la)) )
            r1+=scores[0]['rouge-1']['r']
            r2+=scores[0]['rouge-2']['r']
            rl+=scores[0]['rouge-l']['r']
            num+=1.0
        if num%100==0:
            print(num,r1/num,r2/num,rl/num,0.2*(r1/num)+0.4*(r2/num)+0.4*(rl/num))
    print(len(lables),r1/num,r2/num,rl/num,0.2*(r1/num)+0.4*(r2/num)+0.4*(rl/num))
