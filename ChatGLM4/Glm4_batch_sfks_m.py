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
            if len(answers) > 1:
                option_list = json_line['option_list']
                statement = json_line['statement']
                example1 = '甲公司生产美多牌薰衣草保健枕，美多为注册商标，薰衣草为该枕头的主要原料之一。其产品广告和包装上均突出宣传薰衣草，致使薰衣草保健枕被消费者熟知，其他厂商也推出薰衣草保健枕。后薰衣草被法院认定为驰名商标。下列哪些表述是正确的?A: 甲公司可在一种商品上同时使用两件商标,B: 甲公司对美多享有商标专用权，对薰衣 草不享有商标专用权,C: 法院对驰名商标的认定可写入判决主文,D: 薰衣草叙述了该商品的主要原料，不能申请注册。'
                example1_a = '答案：A、B。'
                example2 = '欣荣公司于2006年8月1日领取营业执照时，股东甲尚有50万元 的出资未缴纳。按照出资协议最晚应于2008年6月1日缴纳，但甲到时仍未缴纳。2011年9月，欣荣公司和公司的债权人乙(债权2009年11月到期)均欲要求甲承担 补足出资的义务。下列哪些选项的说法是正确的?A: 欣荣公司的请求权已经超过诉讼时效, B: 乙的请求权没有超过诉讼时效,C: 欣荣公司的请求权没有超过诉讼时效,D: 乙的请求权已经超过为诉讼时效。'
                example2_a = '答案：B、C。'
                example3 = '在法庭审判过程中，下列哪些情形不可以延期审理?。A:被告人的辩护人申请审判员张某回避,B:被告人收到起诉书后下落不明,C:被告人的辩护人请求重新鉴定,D:用简易程序审理的案件，在法庭审理过程中，发现证据不充分的。'
                example3_a = '答案：A、B、D。'
                example4 = '根据《检察官法》，下列哪些人不得担任检察官?A:赵某，某大学中文系毕业，曾在某公安局工作，后因嫖娼被开除，现已满4年,B:孙某，某大学法律系本科学历，曾在法院工作2年，后下海经商,C:钱某，其父亲是中国人，母亲是美国人，钱某具有美国国籍，但是毕业于中国某大学法律系,D:李某，毕业于某高等专科院校法律系，在一家公司从事法务工作年满5年，去贫困的山区担任检察官。'
                example4_a = '答案：A、C。'
                instruct = "请回答如下多项选择题，有2-4个正确选项，只返回正确选项。"
                s = statement + "A：" + option_list['A'] + "。B：" + option_list['B'] + "。C：" + \
                    option_list['C'] + "D：" + option_list['D']
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
                labels.append(answers)
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
            count += 1
            tem = 0
            for answer in answers:
                if answer in response:
                    tem += 1
            if 'A' in response and  'B' in response and  'C' in response and  'D' in response :
                tem = 4
            if tem == len(answers):
                right += 1
            if count%100==0:
                print(answers, "======", response)
                print(count, right, float(right / count))
    print(count, right, float(right / count))