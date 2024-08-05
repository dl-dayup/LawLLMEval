import torch,json,sys
from transformers import AutoModelForCausalLM, AutoTokenizer

fewshot_num = int(sys.argv[1]) if sys.argv[1] else 0
model_path = './model_yi_6b_chat/'
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype='auto'
).eval()
filename = '../LLMData/按任务打包/事件检测/train0.jsonl'
count = micro_tp = micro_fn = micro_label = micro_p = micro_r = micro_f = macro_f = 0
with (open(filename, 'r', encoding='utf-8') as file):
    for line in file:
        count += 1
        json_line = json.loads(line.strip())
        events = json_line['events']
        content = "".join( i['sentence'] for i in json_line['content'])

        negative_triggers = json_line['negative_triggers']
        answers=set([e['mention'][0]['trigger_word']  for e in events])
        negative_labels = set([n['trigger_word'] for n in negative_triggers])
        s = content
        instruct = "找出给定文本中的事件类型及其触发词。"
        example1 = '徐×1等妨害公务罪一审刑事判决书 妨害公务罪 2014年10月17日23时许，被告人刘×1同刘×2等人将北京市昌平区东小口镇陈营村早市绿化带围挡推倒2014年10月19日11时许，北京市公安局昌平分局民警李×2、杨××等人前往陈营村早市就绿化带围挡被推倒一事进行取证民警在核实被告人刘×1身份时，被告人徐×1、李×1、刘×1拒不配合民警工作，对民警进行言语辱骂、推搡、撕扯，阻扰民警依法执行职务，并造成大量群众围观2014年10月19日，三名被告人被抓获'
        example1_a = '拒、推倒、抓获、撕扯、辱骂、推搡、阻扰'
        example2 = '孙宗富引诱、容留、介绍卖淫一审刑事判决书 容留卖淫罪 2016年7月22日，被告人孙宗富与彭松林（另案处理）承租了大余县南安镇板鸭一条街46号房屋，共同经营梦情休闲沐足店，容留妇女李某1、黄某、陈某1、罗某和赖某在店内卖淫，按照每次卖淫收取398元或698元，店内提成178元或248元收费8月3日，民警查获该休闲店，当场抓获被告人孙宗富、上述五名卖淫人员和二名嫖娼人员，起获店内收入账本、印有孙宗富电话号码的代金券、POS机和避孕套等物品 至案发，该休闲店共非法获利29859元'
        example2_a = '收取、起获、抓获、获利、承租、查获、卖淫、提成、嫖娼'
        example3 = '朱建忠介绍贿赂一审刑事判决书 介绍贿赂罪 2012年至2014年间，蔡某某（另案处理）为使上海XX照明电器有限公司（以下简称“XX公司”）承接到中 山公园地区灯光工程、虹桥古北地区灯光工程，通过被告人朱建忠向时任长宁区绿化和市容管理局局长朱某1（另案处理）请托并许诺给予好处费在XX公司承接 到上述工程后，蔡某某于2012年7月、2014年1月先后两次通过被告人朱建忠给予朱某1贿赂款共计人民币300万元为此，被告人朱建忠从蔡某某获得好处费人民币10万元'
        example3_a = '给予、贿赂款、获得'
        example4 = '汪某拒不支付劳动报酬罪一审刑事判决书 拒不支付劳动报酬罪 2012年7月，被告人汪某租赁汉川市经济开发区汉正服装城E区财富路8路-A号厂房，开办“依米尔豪”服装加工厂（未办理工商登记）2015年8月至2016年1月2日期间，招用李某1、曾某1、周某1等数十名工人进行服装加工，拖欠54名员工劳动报酬404558元后逃匿经汉川市人力社会资源保障局依法下达了限期改正指令书后仍不支付案发后，在提起公诉前，被告人已支付上述全部被拖欠的员工劳动报酬404558元汉川市司法局建议对被告人适用非监禁刑'
        example4_a = '逃匿、支付、招用、建议、租赁、指令'
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
        response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
        # print(answers, response)
        tp = 0
        fn = 0
        for answer in answers:
            if answer in response:
                tp += 1
        for l in negative_labels:
            if l in response:
                fn += 1
        if tp+fn > 0:
            p=tp/(tp+fn)
        else:
            p=0
        r=tp/len(answers)
        if (p+r)==0:
            f=0
        else:
            f=2 * p * r / (p + r)

        macro_f += f
        micro_tp += tp
        micro_fn += fn
        micro_label += len(answers)
        print(count)

micro_p = micro_tp/(micro_tp+micro_fn)
micro_r = micro_tp/micro_label
micro_f = 2*micro_p*micro_r/(micro_p+micro_r)
print(count,micro_f,float(macro_f/count),(micro_f+float(macro_f/count))/2)