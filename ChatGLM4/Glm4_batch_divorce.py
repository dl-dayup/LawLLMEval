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
    batch_message = []
    labels = []
    filename = '../../LLMData/按任务打包/divorce/dev.txt'
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
            statement = json_line['text_a']
            lab = json_line['labels']
            answers = [label_dict[l] for l in lab]
            instruct = "请进行多标签分类，判断上述文本中涉及到以下标签中的哪几个标签？[婚后有子女、限制行为能力子女抚养、有夫妻共同财产、支付抚养费、不动产分割、婚后分居、二次起诉离婚、按月给付抚养费、准予离婚、有夫妻共同债务、婚前个人财产、法定离婚、不履行家庭义务、存在非婚生子、适当帮助、不履行离婚协议、损害赔偿、感情不和分居满二年、子女随非抚养权人生活、婚后个人财产]"
            example1 = '被告从2010年开始沉迷赌博，对家庭不管不顾，从2012年开始便离家出走，至今下落不明。'
            example1_a = '不履行家庭义务、感情不和分居满二年'
            example2 = '2010年3月初，原告为了与我离婚，抛下女儿，将家里的现金全部带走，与我们父女俩分居生活，弄得我身无分文，为了生活不得已将帕萨特牌轿 车卖给了朋友葛峥伟，获得车价款15万元，车辆已办理过户。'
            example2_a = '不履行家庭义务、婚后分居、婚后有子女、有夫妻共同财产'
            example3 = '2、判令被告给付女儿抚养费人民币60000元（自2012年1月至2017年1月共5年计60个月每月1000元）；'
            example3_a = '婚后有子女、支付抚养费、限制行为能力子女抚养'
            example4 = '二、婚生一女方某由女方扶养到独立生活，男方每月给抚养费叁仟元到孩子大学毕业，男方有探视权；'
            example4_a = '婚后有子女、按月给付抚养费、支付抚养费、限制行为能力子女抚养'
            s = statement
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
        for response,answers in zip(batch_responses,bls):
            count+=1
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
                print(answers, "======", response)
                print(count, float(macro_f / count))

    micro_p = micro_tp / micro_predict
    micro_r = micro_tp / micro_label
    micro_f = 2 * micro_p * micro_r / (micro_p + micro_r)
    print(count, micro_f, float(macro_f / count), (micro_f + float(macro_f / count)) / 2)