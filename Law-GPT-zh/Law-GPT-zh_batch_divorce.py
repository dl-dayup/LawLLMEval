import torch,json
from transformers import AutoTokenizer, AutoModel
from model.modeling_chatglm import ChatGLMForConditionalGeneration

tokenizer = AutoTokenizer.from_pretrained("model", trust_remote_code=True)
model_class = ChatGLMForConditionalGeneration
model = model_class.from_pretrained("model", device_map = "auto").half().eval()

filename = '../../LLMData/按任务打包/divorce/dev.txt'
label_dict= {0:"婚后有子女", 1:"限制行为能力子女抚养", 2:"有夫妻共同财产", 3:"支付抚养费", 4:"不动产分割", 5:"婚后分居", 6:"二次起诉离婚", 7:"按月给付抚养费", 8:"准予离婚",
            9:"有夫妻共同债务", 10:"婚前个人财产", 11:"法定离婚", 12:"不履行家庭义务", 13:"存在非婚生子", 14:"适当帮助", 15:"不履行离婚协议", 16:"损害赔偿", 17:"感情不和分居满二年",
            18:"子女随非抚养权人生活", 19:"婚后个人财产"}
all_labels=label_dict.values()
count = micro_tp = micro_predict = micro_label = micro_p = micro_r = micro_f = macro_f = 0
with (open(filename, 'r', encoding='utf-8') as file):
    for line in file:
        json_line = json.loads(line.strip())
        count+=1
        statement = json_line['text_a']
        labels = json_line['labels']
        answers = [label_dict[l] for l in labels]
        # example1 = '参考如下例子回答多项选择题，需要返回所有满足条件的选项。例子1：甲公司生产美多牌薰衣草保健枕，美多为注册商标，薰衣草为该枕头的主要原料之一。其产品广告和包装上均突出宣传薰衣草，致使薰衣草保健枕被消费者熟知，其他厂商也推出薰衣草保健枕。后薰衣草被法院认定为驰名商标。下列哪些表述是正确的?A: 甲公司可在一种商品上同时使用两件商标,B: 甲公司对美多享有商标专用权，对薰衣 草不享有商标专用权,C: 法院对驰名商标的认定可写入判决主文,D: 薰衣草叙述了该商品的主要原料，不能申请注册。答案：A、B。'
        # example2 = '例子2：欣荣公司于2006年8月1日领取营业执照时，股东甲尚有50万元 的出资未缴纳。按照出资协议最晚应于2008年6月1日缴纳，但甲到时仍未缴纳。2011年9月，欣荣公司和公司的债权人乙(债权2009年11月到期)均欲要求甲承担 补足出资的义务。下列哪些选项的说法是正确的?A: 欣荣公司的请求权已经超过诉讼时效, B: 乙的请求权没有超过诉讼时效,C: 欣荣公司的请求权没有超过诉讼时效,D: 乙的请求权已经超过为诉讼时效。答案：B、C。'
        # example3 = '例子3：在法庭审判过程中，下列哪些情形不可以延期审理?。A:被告人的辩护人申请审判员张某回避,B:被告人收到起诉书后下落不明,C:被告人的辩护人请求重新鉴定,D:用简易程序审理的案件，在法庭审理过程中，发现证据不充分的。答案：A、B、D。'
        # example4 = '例子4：根据《检察官法》，下列哪些人不得担任检察官?A:赵某，某大学中文系毕业，曾在某公安局工作，后因嫖娼被开除，现已满4年,B:孙某，某大学法律系本科学历，曾在法院工作2年，后下海经商,C:钱某，其父亲是中国人，母亲是美国人，钱某具有美国国籍，但是毕业于中国某大学法律系,D:李某，毕业于某高等专科院校法律系，在一家公司从事法务工作年满5年，去贫困的山区担任检察官。答案：A、C。'
        # example = ''+example1+example2+example3+example4
        s = statement + "请进行多标签分类，判断上述文本中涉及到以下标签中的哪几个标签？[婚后有子女、限制行为能力子女抚养、有夫妻共同财产、支付抚养费、不动产分割、婚后分居、二次起诉离婚、按月给付抚养费、准予离婚、有夫妻共同债务、婚前个人财产、法定离婚、不履行家庭义务、存在非婚生子、适当帮助、不履行离婚协议、损害赔偿、感情不和分居满二年、子女随非抚养权人生活、婚后个人财产]"
        response = model.chat(tokenizer, s[:4096], history='', max_length=4096)
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
        if predict>0:
            p=tp/predict
        else:
            p=0
        r=tp/len(answers)
        if (p+r)==0:
            f=0
        else:
            f=2 * p * r / (p + r)

        macro_f += f
        print(count)

micro_p = micro_tp/micro_predict
micro_r = micro_tp/micro_label
micro_f = 2*micro_p*micro_r/(micro_p+micro_r)
print(count,micro_f,float(macro_f/count),(micro_f+float(macro_f/count))/2)