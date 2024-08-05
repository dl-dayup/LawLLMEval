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
filename = '../LLMData/按任务打包/司法考试/0_train0.json'
count=num=right=dlen=0
with open(filename, 'r', encoding='utf-8') as file:
    for line in file:
        json_line = json.loads(line.strip())
        answers = json_line['answer']
        if len(answers) > 1:
            count += 1
            option_list = json_line['option_list']
            statement = json_line['statement']
            example1 = '甲公司生产美多牌薰衣草保健枕，美多为注册商标，薰衣草为该枕头的主要原料之一。其产品广告和包装上均突出宣传薰衣草，致使薰衣草保健枕被消费者熟知，其他厂商也推出薰衣草保健枕。后薰衣草被法院认定为驰名商标。下列哪些表述是正确的?A: 甲公司可在一种商品上同时使用两件商标,B: 甲公司对美多享有商标专用权，对薰衣 草不享有商标专用权,C: 法院对驰名商标的认定可写入判决主文,D: 薰衣草叙述了该商品的主要原料，不能申请注册。'
            example1_a = '答案：A、B。'
            example2 = '欣荣公司于2006年8月1日领取营业执照时，股东甲尚有50万元的出资未缴纳。按照出资协议最晚应于2008年6月1日缴纳，但甲到时仍未缴纳。2011年9月，欣荣公司和公司的债权人乙(债权2009年11月到期)均欲要求甲承担 补足出资的义务。下列哪些选项的说法是正确的?A: 欣荣公司的请求权已经超过诉讼时效, B: 乙的请求权没有超过诉讼时效,C: 欣荣公司的请求权没有超过诉讼时效,D: 乙的请求权已经超过为诉讼时效。'
            example2_a = '答案：B、C。'
            example3 = '在法庭审判过程中，下列哪些情形不可以延期审理?。A:被告人的辩护人申请审判员张某回避,B:被告人收到起诉书后下落不明,C:被告人的辩护人请求重新鉴定,D:用简易程序审理的案件，在法庭审理过程中，发现证据不充分的。'
            example3_a = '答案：A、B、D。'
            example4 = '根据《检察官法》，下列哪些人不得担任检察官?A:赵某，某大学中文系毕业，曾在某公安局工作，后因嫖娼被开除，现已满4年,B:孙某，某大学法律系本科学历，曾在法院工作2年，后下海经商,C:钱某，其父亲是中国人，母亲是美国人，钱某具有美国国籍，但是毕业于中国某大学法律系,D:李某，毕业于某高等专科院校法律系，在一家公司从事法务工作年满5年，去贫困的山区担任检察官。'
            example4_a = '答案：A、C。'
            instruct = "请回答如下多项选择题，有2-4个正确选项，只返回正确选项。"
            s = statement + "A：" + option_list['A'] + "。B：" + option_list['B'] + "。C：" + \
                option_list['C'] + "D：" + option_list['D']
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
            input_ids = tokenizer.apply_chat_template(conversation=messages, tokenize=True, add_generation_prompt=True, return_tensors='pt')
            output_ids = model.generate(input_ids.to('cuda'))
            response = tokenizer.decode(output_ids[0][input_ids.shape[1]:], skip_special_tokens=True)
            # print(answers, "======", response)
            tem = 0
            for answer in answers:
                if answer in response:
                    tem += 1
            if 'A' in response and 'B' in response and 'C' in response and 'D' in response:
                tem = 4
            if tem == len(answers):
                right += 1
            print(count, right, float(right / count))
print(count, right, float(right / count))
