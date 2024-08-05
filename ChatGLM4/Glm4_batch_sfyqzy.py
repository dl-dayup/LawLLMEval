import json,sys,jieba
from rouge_chinese import Rouge
from typing import Optional, Union
from transformers import AutoModel, AutoTokenizer, LogitsProcessorList

fewshot_num = int(sys.argv[1]) if sys.argv[1] else 0
MODEL_PATH = '../../glm-4-9b-chat'
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    encode_special_tokens=True)
model = AutoModel.from_pretrained(MODEL_PATH, trust_remote_code=True, device_map="auto").eval()
def process_model_outputs(inputs, outputs, tokenizer):
    responses = []
    for input_ids, output_ids in zip(inputs.input_ids, outputs):
        response = tokenizer.decode(output_ids[len(input_ids):], skip_special_tokens=True).strip()
        responses.append(response)
    return responses

def batch(
        model,
        tokenizer,
        messages: Union[str, list[str]],
        max_input_tokens: int = 8192,
        max_new_tokens: int = 8192,
        num_beams: int = 1,
        do_sample: bool = True,
        top_p: float = 0.8,
        temperature: float = 0.8,
        logits_processor: Optional[LogitsProcessorList] = LogitsProcessorList(),
):
    messages = [messages] if isinstance(messages, str) else messages
    batched_inputs = tokenizer(messages, return_tensors="pt", padding="max_length", truncation=True,
                               max_length=max_input_tokens).to(model.device)
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "num_beams": num_beams,
        "do_sample": do_sample,
        "top_p": top_p,
        "temperature": temperature,
        "logits_processor": logits_processor,
        "eos_token_id": model.config.eos_token_id
    }
    batched_outputs = model.generate(**batched_inputs, **gen_kwargs)
    batched_response = process_model_outputs(batched_inputs, batched_outputs, tokenizer)
    return batched_response

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
    batch_message = []
    labels = []
    filename = '../../LLMData/按任务打包/涉法舆情摘要/train0.jsonl'
    rouge = Rouge()
    count = rouge1_f = rouge2_f = rougel_f = 0
    with (open(filename, 'r', encoding='utf-8') as file):
        for line in file:
            json_line = json.loads(line.strip())
            statement = json_line['text']
            summary = json_line['summary']
            instruct = "请返回上述文本相应的摘要。"
            s = statement
            messages = [{"role": "system", "content": instruct}, {"role": "user", "content": s[:3000]}]
            labels.append(summary)
            batch_message.append(messages)
    split_result = split_lists(batch_message, labels)
    for i, (bqs, bls) in enumerate(split_result):
        batch_inputs = []
        max_input_tokens = 4096
        for i, messages in enumerate(bqs):
            new_batch_input = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            max_input_tokens = max(max_input_tokens, len(new_batch_input))
            batch_inputs.append(new_batch_input)
        gen_kwargs = {
            "max_input_tokens": max_input_tokens,
            "max_new_tokens": 200,
            "do_sample": True,
            "top_p": 0.8,
            "temperature": 0.8,
            "num_beams": 1,
        }

        batch_responses = batch(model, tokenizer, batch_inputs, **gen_kwargs)
        for response,summary in zip(batch_responses,bls):
            count+=1
            scores = rouge.get_scores(' '.join(jieba.cut(response)), ' '.join(jieba.cut(summary)))
            rouge1_f += scores[0]['rouge-1']['f']
            rouge2_f += scores[0]['rouge-2']['f']
            rougel_f += scores[0]['rouge-l']['f']
            print(count, rouge1_f, rouge2_f, rougel_f)

    print(count, float(rouge1_f / count), float(rouge2_f / count), float(rougel_f / count),
                              0.2 * float(rouge1_f / count) + 0.4 * float(rouge2_f / count) + 0.4 * float(rougel_f / count))