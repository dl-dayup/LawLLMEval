import json,sys
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
    filename = '../../LLMData/按任务打包/司法考试/0_train.json'
    count = num = right = 0
    batch_message = []
    labels = []
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            json_line = json.loads(line.strip())
            answers = json_line['answer']
            if len(answers) == 1:
                answer = answers[0]
                option_list = json_line['option_list']
                statement = json_line['statement']
                instruct = "请回答如下单项选择题，只需返回一个正确选项。"
                example1 = '以下有关民事简易程序的说法中，正确的有?A：简易程序的审限为三个月，属于法定不可变期间。B：法院在审理案件过程中发现案情复杂，需要转为普通程序的，应当在审限届满前作出决定并通知当事人。C：简易程序的开庭方式较为灵活便捷，经过双方当事人同意，法院可以采用同步视频等视听传输技术的方式开庭。D：适用简易程序案件的举证期限可以由法院确定，也可由当事人协商并经法院准许，但不得超过30日.'
                example1_a = '答案：C。'
                example2 = '下列关于自然人民事权利能力和民事行为能力法律适用的说法，符合我国《涉外民事关系法律适用法》规定的是哪一项？A：自然人的民事权利能力，适用经 常居所地法律，但涉及婚姻家庭、继承的除外。B：自然人从事民事活动，依照经常居所地法律为无民事行为能力，依照行为地法律为有民事行为能力的，一律适用行为地法律。C：自然人的民事权利能力，一律适用经常居所地法律。D：自然人的民事行为能力，一律适用经常居所地法律。'
                example2_a = '答案：C。'
                example3 = '行政机关对于重大违法行为给予较重的行政处罚时，在证据可能灭失的情况下，下列选项哪个是正确的?A：经行政机关负责人批准，可以先行封存证据。B：经行政机关集体讨论决定，可以先行扣押 证据。C：经行政机关负责人批准，可以先行登记保存证据。D：经行政机关负责人批准，可以先行登记提存证据。'
                example3_a = '答案：C。'
                example4 = '根据《海牙规则》，下列事项中承运人可以主张免责的是哪一项？A：由于积载不当导致的货物损失。B：由于开航前船舶装备不足以抵抗预定航线上的一般风险 导致船货沉没的损失。C：航行过程中由于承运人判断严重失误而点火烘烤发动机引起火灾造成的货损。D：由于船舶救助海上遇难其他船舶而发生绕航对货物造成的损失。'
                example4_a = '答案：D。'
                s = statement + "。A：" + option_list['A'] + "。B：" + option_list['B'] + "。C：" + option_list[
                    'C'] + "。D：" + option_list['D']
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
                labels.append(answer)
                batch_message.append(messages)
    split_result = split_lists(batch_message, labels)
    for i, (bqs, bls) in enumerate(split_result):
        batch_inputs = []
        max_input_tokens = 1024
        for i, messages in enumerate(bqs):
            new_batch_input = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            max_input_tokens = max(max_input_tokens, len(new_batch_input))
            batch_inputs.append(new_batch_input)
        gen_kwargs = {
            "max_input_tokens": max_input_tokens,
            "max_new_tokens": 20,
            "do_sample": True,
            "top_p": 0.8,
            "temperature": 0.8,
            "num_beams": 1,
        }

        batch_responses = batch(model, tokenizer, batch_inputs, **gen_kwargs)
        for response,label in zip(batch_responses,bls):
            count+=1
            if answer in response:
                right+=1
            if 'A' in response and  'B' in response and  'C' in response and  'D' in response and '答案：'+answer not in response:
                right-=1
            if count%100==0:
                print(answer, "======", response)
                print(count, right, float(right / count))
    print(count, right, float(right / count))