import torch,json,jieba,sys
from rouge_chinese import Rouge
from transformers import AutoModelForCausalLM, AutoTokenizer

fewshot_num = int(sys.argv[1]) if sys.argv[1] else 0
model_path = './model_yi_6b_chat/'
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype='auto'
).eval()
filename = '../LLMData/按任务打包/司法摘要/data_second_train0.json'

rouge=Rouge()
count = rouge1_f = rouge2_f = rougel_f = t_len = s_len = 0
with (open(filename, 'r', encoding='utf-8') as file):
    for line in file:
        json_line = json.loads(line.strip())
        count+=1
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
        messages = []
        messages.append({"role": "system", "content": instruct})
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
        input_ids = tokenizer.apply_chat_template(conversation=messages, tokenize=True, add_generation_prompt=True,
                                                  return_tensors='pt')
        output_ids = model.generate(input_ids.to('cuda'))
        response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True).replace('\n','')
        # print(response, '##############', summary)

        scores = rouge.get_scores(' '.join(jieba.cut(response)), ' '.join(jieba.cut(summary)))
        # print(scores)
        rouge1_f += scores[0]['rouge-1']['f']
        rouge2_f += scores[0]['rouge-2']['f']
        rougel_f += scores[0]['rouge-l']['f']
        print(count, rouge1_f,rouge2_f,rougel_f)

print(count, float(rouge1_f/count), float(rouge2_f/count), float(rougel_f/count), t_len/count, s_len/count)