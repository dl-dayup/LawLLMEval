import torch,json
from transformers import AutoTokenizer, AutoModel
from model.modeling_chatglm import ChatGLMForConditionalGeneration

tokenizer = AutoTokenizer.from_pretrained("model", trust_remote_code=True)
model_class = ChatGLMForConditionalGeneration
model = model_class.from_pretrained("model", device_map = "auto").half().eval()

filename = '../../LLMData/按任务打包/事件检测/train0.jsonl'
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
        # print('content:',json_line['title'],json_line['crime'],content)
        # print('events:',labels)
        # print('negative_triggers:',negative_labels)

        # example1 = '参考如下例子回答多项选择题，需要返回所有满足条件的选项。例子1：甲公司生产美多牌薰衣草保健枕，美多为注册商标，薰衣草为该枕头的主要原料之一。其产品广告和包装上均突出宣传薰衣草，致使薰衣草保健枕被消费者熟知，其他厂商也推出薰衣草保健枕。后薰衣草被法院认定为驰名商标。下列哪些表述是正确的?A: 甲公司可在一种商品上同时使用两件商标,B: 甲公司对美多享有商标专用权，对薰衣 草不享有商标专用权,C: 法院对驰名商标的认定可写入判决主文,D: 薰衣草叙述了该商品的主要原料，不能申请注册。答案：A、B。'
        # example2 = '例子2：欣荣公司于2006年8月1日领取营业执照时，股东甲尚有50万元 的出资未缴纳。按照出资协议最晚应于2008年6月1日缴纳，但甲到时仍未缴纳。2011年9月，欣荣公司和公司的债权人乙(债权2009年11月到期)均欲要求甲承担 补足出资的义务。下列哪些选项的说法是正确的?A: 欣荣公司的请求权已经超过诉讼时效, B: 乙的请求权没有超过诉讼时效,C: 欣荣公司的请求权没有超过诉讼时效,D: 乙的请求权已经超过为诉讼时效。答案：B、C。'
        # example3 = '例子3：在法庭审判过程中，下列哪些情形不可以延期审理?。A:被告人的辩护人申请审判员张某回避,B:被告人收到起诉书后下落不明,C:被告人的辩护人请求重新鉴定,D:用简易程序审理的案件，在法庭审理过程中，发现证据不充分的。答案：A、B、D。'
        # example4 = '例子4：根据《检察官法》，下列哪些人不得担任检察官?A:赵某，某大学中文系毕业，曾在某公安局工作，后因嫖娼被开除，现已满4年,B:孙某，某大学法律系本科学历，曾在法院工作2年，后下海经商,C:钱某，其父亲是中国人，母亲是美国人，钱某具有美国国籍，但是毕业于中国某大学法律系,D:李某，毕业于某高等专科院校法律系，在一家公司从事法务工作年满5年，去贫困的山区担任检察官。答案：A、C。'
        # example = ''+example1+example2+example3+example4
        s = content + "。找出这段话中的事件类型及其触发词。"
        response = model.chat(tokenizer, s[:4096], history='', max_length=4096)
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