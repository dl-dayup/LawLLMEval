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
    dict1={}
    dict2={}
    filename = './divorce/train.txt'
    f1 = open('./divorce/tags.txt', "r", encoding='utf-8')
    index =0
    for line in f1:
        dict1[index]=line.strip()
        index+=1
    f2 = open('./divorce/selectedtags.txt', "r", encoding='utf-8')
    index =0
    for line in f2:
        dict2[line.strip()]=index
        index+=1
    with open(filename, 'r', encoding='utf-8') as file:
            lines=file.readlines()
            for line in lines:
              data = json.loads(line.strip())
              out_text = data['text_a']
              if len(out_text)<280:
                correct_answer = data['labels']
                lables.append([dict1[i] for i in correct_answer]) 
                batch_queries.append( "<|user|>\n"+out_text+"请进行多标签分类，谨慎思考，避免输出过多标签，判断上述文本中涉及到以下标签中的哪一个或几个标签？婚后有子女、限制行为能力子女抚养、有夫妻共同财产、支付抚养费、不动产分割、婚后分居、二次起诉离婚、按月给付抚养费、准予离婚、有夫妻共同债务、婚前个人财产、法定离婚、不履行家庭义务、存在非婚生子、适当帮助、不履行离婚协议、损害赔偿、感情不和分居满二年、子女随非抚养权人生活、婚后个人财产。[dict1[i] for i in correct_answer])\n<|assistant|>")
    split_result = split_lists(batch_queries, lables)
    for i, (bqs, bls) in enumerate(split_result):
        print("=batch" * 10,i)
        batch_responses = main(bqs)
        for response,label in zip(batch_responses,bls):
            predict=[]
            print("=" * 10,label)
            print(response) #,response.index(label.strip()))
            for item in dict2.keys():
                if item in response:
                    predict.append(dict1[dict2[item]])
            predicts.append(predict)
    print(len(lables),evaluate(predicts,lables,'./divorce/tags.txt'))
