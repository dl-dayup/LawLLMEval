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
    filename = '../../LLMData/按任务打包/论辩理解/stage_1/train_entry0.jsonl'
    count = num = right = dlen = 0
    with (open(filename, 'r', encoding='utf-8') as file):
        for line in file:
            json_line = json.loads(line.strip())
            sc = json_line['sc']
            crime = json_line['crime']
            bc_1 = json_line['bc_1']
            bc_2 = json_line['bc_2']
            bc_3 = json_line['bc_3']
            bc_4 = json_line['bc_4']
            bc_5 = json_line['bc_5']
            label = json_line['answer']
            instruct = "以下五个句子中有一个句子可以与给定文本能组成有争议的观点对，请判断是哪一个句子并返回句子号。不需要重复原文内容和给定的五个句子内容。"
            example1 = '原告人诉称，时间时间时间21时许，杜某携带家属在盘县松河乡洪发酒楼二楼KTV为其女儿过生日，期间，管某进入包房敬酒，后杜某送管某出包房，与管某 同行的沙某认为杜某不尊重管某，遂与杜某发生肢体冲突。杜某、李某开车到坪地乡邀约浦某、李某等人前往松某乡，到松某乡后，杜某、李某将事先准备好的刀和钢管分发给浦某、刘某由等人，发完刀后，杜某吩咐浦某等人原地等候，并约定以吼声为暗号，浦某等人听到杜某的暗号后，持刀前往洪发酒店，并在酒店门口与管某等人发生斗殴，斗殴过程中，管某被杜某、李某等人持刀砍杀数十刀后死亡。bc_1：其辩护人提出被告人黄某仅提着自己的刀站在洪发酒店的斜对面，未进入打架现场参与斗殴，没有聚众斗殴的行为；其未组织或者邀约他人聚众斗殴，不是本案的组织者或积极参加者，不符合聚众斗殴罪的犯罪主体；黄某不认识对方人员，没有形成与对方斗殴的主观故意；本案的危害结果与被告人黄某的行为无直接的因果关系，综上，起诉书指控被告人黄某构成聚众斗殴罪的指控不能成立，请求法院宣告被告人黄某无罪的辩护意见。bc_2：被告人谢某对起诉书指控的事实及罪名无异议。bc_3：被告人黄某对起诉书指控的事实及罪名无异议。bc_4：黄某对起诉书指控的事实及罪名无异议。bc_5：被告人刘某由提出其仅参与打架，但未伤害对方人员，其行为不构成故意伤害罪，应构成聚众斗殴罪的辩解意见。'
            example1_a = '4'
            example2 = '现要求依法追究被告人谢某甲的刑事责任；责令被告人赔偿其残疾赔偿金、医疗费、误工费、护理费、住院伙食补助费、营养费、交通费、鉴定费、后续治疗费等共计人民币103059元。bc_1：其是受害人，对方是肇事者，应承担责任，其不承担责任，对民事部分也不承担赔偿责任。辩护人的辩护意见：1、本案证据 不能证明樊某甲身上的伤是被告人谢某甲造成的；2、被告人谢某甲行为属正当防卫；3、樊某甲等三人的行为构成非法侵入住宅；4、被告人谢某甲无罪。bc_2 ：辩护人的辩护意见：1、本案证据不能证明樊某甲身上的伤是被告人谢某甲造成的；2、被告人谢某甲行为属正当防卫；3、樊某甲等三人的行为构成非法侵入 住宅；4、被告人谢某甲无罪。bc_3：其楼梯上还没有下来就遭到对方三人用东西及拳头殴打，其无奈拿起菜刀还击，随意挥舞。bc_4：其是受害人，对方是肇 事者，应承担责任。bc_5：被告人谢某甲辩称，其是听到玻璃门推到、其妻子被对方打了之后才从楼上下来的，其看到对方人多，其孙女就拨打了110报警电话 。'
            example2_a = '1'
            example3 = '附带民事诉讼原告人樊某甲诉称，时间时间时间17时许，为给外甥准备第二天的校服和生活用品，其和女儿徐某乙、徐某丙一同来到新桥街闹市巷3-2-101室 。三人还未踏上地下改装的门厅楼梯，就被等候在楼梯台阶上的谢某甲用早已准备好的刀砍到，致其前额骨骨折，左右眼角边、右脸及左右手共被砍大小九处，住院治疗17天、在家治养28天。bc_1：辩护人的辩护意见：1、本案证据不能证明樊某甲身上的伤是被告人谢某甲造成的；2、被告人谢某甲行为属正当防卫；3 、樊某甲等三人的行为构成非法侵入住宅；4、被告人谢某甲无罪。bc_2：其是受害人，对方是肇事者，应承担责任。bc_3： 被告人谢某甲辩称，其是听到玻璃门推到、其妻子被对方打了之后才从楼上下来的，其看到对方人多，其孙女就拨打了110报警电话。其楼梯上还没有下来就遭到对方三人用东西及拳头殴打，其 无奈拿起菜刀还击，随意挥舞。bc_4：其楼梯上还没有下来就遭到对方三人用东西及拳头殴打，其无奈拿起菜刀还击，随意挥舞。bc_5：其不承担责任，对民事部分也不承担赔偿责任。'
            example3_a = '3'
            example4 = '某甲用早已准备好的刀砍到，致其前额骨骨折，左右眼角边、右脸及左右手共被砍大小九处，住院治疗17天、在家治养28天。所受损伤评定为十级伤残。bc_1：被告人谢某甲辩称，其是听到玻璃门推到、其妻子被对方打了之后才从楼上下来的，其看到对方人多，其孙女就拨打了110报警电话。bc_2：其不承担责任， 对民事部分也不承担赔偿责任。bc_3：其是受害人，对方是肇事者，应承担责任。bc_4：辩护人的辩护意见：1、本案证据不能证明樊某甲身上的伤是被告人谢 某甲造成的；2、被告人谢某甲行为属正当防卫；3、樊某甲等三人的行为构成非法侵入住宅；4、被告人谢某甲无罪。bc_5：其楼梯上还没有下来就遭到对方三 人用东西及拳头殴打，其无奈拿起菜刀还击，随意挥舞。'
            example4_a = '5'
            s = sc + "bc_1：" + bc_1 + "。bc_2：" + bc_2 + "。bc_3：" + bc_3 + "。bc_4: " + bc_4 + "。bc_5: " + bc_5
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
            labels.append(label)
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
            "max_new_tokens": 20,
            "do_sample": True,
            "top_p": 0.8,
            "temperature": 0.8,
            "num_beams": 1,
        }
        batch_responses = batch(model, tokenizer, batch_inputs, **gen_kwargs)
        for response,label in zip(batch_responses,bls):
            count+=1
            if str(label) in response:
                right += 1
            if '1' in response and '2' in response and '答案：' + str(label) not in response:
                right -= 1
            if count % 100 == 0:
                print(count, label, '=============', response)
                print(count, right, float(right / count))
    print(count, right, float(right / count))