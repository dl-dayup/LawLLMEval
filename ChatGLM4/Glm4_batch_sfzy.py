import json,jieba,sys
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
    filename = '../../LLMData/按任务打包/司法摘要/data_second_train0.json'

    rouge = Rouge()
    count = rouge1_f = rouge2_f = rougel_f = t_len = s_len = 0
    with (open(filename, 'r', encoding='utf-8') as file):
        for line in file:
            json_line = json.loads(line.strip())
            statement = json_line['question']
            candidates = json_line['candidates']
            candi = "。".join(ca for ca in candidates)
            summary = json_line['answer']
            instruct = "给定问题和参考文本，你需要给出问题的六十字左右的回答。"
            example1 = '问题：你好，我在X上班，老婆在XX上班，分开X年了，现在她提出离婚。参考文本：对方提出离婚的，积极去应诉处理吧。离婚你要去应诉。委托律师处理。协商处理。'
            example1_a = '你好，对方提出离婚的，积极去应诉处理。'
            example2 = '问题：行人全程不看马路玩手机背对行驶车辆过马路全程都没有看过一眼车道有监控汽车速度已经减下来避让不及轻微碰怎么划分？参考文本：报警，由交警部门出具的事故责任认定书划分责任..。事故责任认定书是怎么划分事故责任的，如果承担主 要责任以上责任，是需要承担刑事责任的。您好，停车的一方有责任，但具体责任划分由交警队进行交通事故责任认定。'
            example2_a = '你好，报警，由交警部门出具的事故责任认定书划分责任。'
            example3 = '问题：我被骗通过xx教育机构，在xx分期借贷XXXXX，身份证已实名，订单已产生，银行卡未绑定，请问是否成功申请贷款，且需要还款吗？参考文本：查看合同违约责任即可。结合当初约定以及行为合法性情况确定。建议直接联系律师商谈，以便更好的为你全面分析和详细解答。'
            example3_a = '您好，如果有证据证明是被骗，可以主张撤销合同。'
            example4 = '问题：在XX美上班，工资已拖欠了X个月，可不可以报警？参考文本：属于民事纠纷，报警没有实际意义，建议向劳动监察大队投诉或申请劳动仲裁。这个属于劳动纠纷，报警没有意义的，你只有申请劳动仲裁进行维权。这个属于劳动纠纷，报警没有意义的，你只有申请劳动仲裁进行维权。这种情况可以到当地的人社局投诉也可以申请劳动仲裁维权。'
            example4_a = '你好，这个属于劳动纠纷， 报警没有意义的，这种情况可以到当地的人社局投诉也可以申请劳动仲裁维权。'
            s = '问题：' + statement + '。参考文本：' + candi
            messages = [{"role": "system", "content": instruct}]
            if fewshot_num >= 1:
                messages.append({"role": "user", "content": example1})
                messages.append({"role": "assistant", "content": example1_a})
            if fewshot_num >= 2:
                messages.append({"role": "user", "content": example2})
                messages.append({"role": "assistant", "content": example2_a})
            if fewshot_num >= 3:
                messages.append({"role": "user", "content": example3})
                messages.append({"role": "assistant", "content": example3_a})
            if fewshot_num >= 4:
                messages.append({"role": "user", "content": example4})
                messages.append({"role": "assistant", "content": example4_a})
            messages.append({"role": "user", "content": s})
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
            "max_new_tokens": 100,
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
            if count % 100 == 0:
                print(count, summary, '=============', response)
                print(count, float(rouge1_f / count), float(rouge2_f / count), float(rougel_f / count),
                      0.2 * float(rouge1_f / count) + 0.4 * float(rouge2_f / count) + 0.4 * float(rougel_f / count))
    print(count, float(rouge1_f / count), float(rouge2_f / count), float(rougel_f / count),
              0.2 * float(rouge1_f / count) + 0.4 * float(rouge2_f / count) + 0.4 * float(rougel_f / count))