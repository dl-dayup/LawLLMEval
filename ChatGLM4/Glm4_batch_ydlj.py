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
    filename = '../../LLMData/按任务打包/阅读理解/ydlj_train.json'
    count = num = right = num1 = 0
    with open(filename, 'r', encoding='utf-8') as file:
        data = json.load(file)['data']
        for d in data:
            p = d['paragraphs'][0]
            context = p['context']
            casename = p['casename']
            qas = p['qas'][0]
            question = qas['question']
            answers_q = qas['answers'][0]
            answers = []
            for i in answers_q:
                answers.append(i['text'])
            example = ''
            # s = example + context + '请问在这个'+casename+'中，'+question+'，如果文中没有提及请说无法回答'
            s = context + question
            instruct = "参考给定内容，回答提出问题，如果文中没有提及请说无法回答"
            example1 = '经审理查明,原告马某于1970年至1986年期间在灌云县交通运输局东辛交管站工作,系平车工人。2011年12月,根据国家和江苏省关于解 决未参保城镇企业退休人员基本养老保障等遗留问题的相关政策,原告一次性缴纳养老保险费用53590元,纳入城镇企业职工养老保险。被告市人社局于同月为原 告办理了退休审批手续,为原告核发了退休证。因原告及其原工作单位灌云交通局东辛交管站未能提供其在该单位工作期间的原始档案材料,被告未对其上述工作期间认定为可视同缴费年限的连续工龄,被告根据相关政策核定原告退休时的养老金金额为646.30元。另查明,案外人于宝华与原告同期办理相关退休手续。于宝华职工档案中的集体所有制单位职工登记表显示,其于1970年8月参加工作,于1986年9月5日经原灌云县劳动局核准属县属集体固定工。档案中的集体职工工资待 遇评定表显示确定计算连续工龄时间为1970年8月,核定为4级工。被告市人社局根据上述材料认定于保华1986年12月31日前可以视同缴费年限的连续工龄为16年5个月,核定的退休金金额为1173.50元。原告认为其1970年至1986年在灌云交通局东辛交管站工作期间应当认定为视同缴费年限的连续工龄,并提供第三人灌云交 通局保存的1979年各公社镇交管站职工名册中东辛交管站职工情况统计表封面(第164页),职工情况登记表(第170页),证实原告马某不属于临时工,并据此于2017 年6月向被告市人社局申请对其重新认定连续工龄,被告市人社局于6月15日作出《关于重新认定连续工龄申请的答复》,告知原告其提供的现有材料不能作为认定连续工龄的有效证明,对其要求重新认定连续工龄的申请不予支持。原告不服向本院提起诉讼。人社局认定马某和于宝华的退休金金额分别为多少？'
            example1_a = '646.30元和1173.50元'
            example2 = '根据当事人陈述和经审查确认的证据,本院认定事实如下:2013年4月8日,阙x9与岳x4签订船舶买卖合同,约定岳x4以396万元的价格购买 阙x9所属“渝发698”轮。2013年5月28日,岳x4与江7公司签订船舶经营挂靠合同,约定岳x4将其所有的“江7801”轮以产权和经营权挂靠在江7公司名下的方式从事经营,岳x4应向江7公司交纳挂靠费为15000元/年,挂靠期为5年,从2013年6月1日起至2018年5月31日止。挂靠期间船舶的经营和管理事宜由岳x4全部自行负责,所发 生的一切费用也由岳x4全部负责支付。2016年12月30日,重庆市港航管理局颁发的船舶营业运输证记载,“江7801”轮曾用名“渝发698”,船舶所有人、经营人均为江7公司。2017年10月29日,岳x4出具欠条,载明欠阎x0工资33000元。该欠条同时加盖有“江7801号业务专用章”。2018年5月2日,阎x0等人向万州仲裁委申请仲裁,请求裁决江7公司支付工资及拖欠劳动报酬经济补偿金,万州仲裁委于当日作出万州劳人仲案不字[2018]第57-61号不予受理通知书,以阎x0请求的事项不属于劳动仲裁受理范围为由,决定对阎x0的仲裁申请不予受理。冉华荣(曾任“江7801”轮船长)在江7公司微信工作群中,阎x0不在该群中,该群群主、管理员系江7公司办公室 主任付仕伟,群文件包括付仕伟等人分享的行政机关事故通报、通告、安全隐患专项排查表、船员考试计划、船舶检验申请书、内河船舶船员值班规则、气象信 息、通知等,群聊天内容主要系船舶装货、航行、卸货具体情况等。“渝发698”轮所属人以什么价格将此出售给岳x4以及岳x4何时签订挂靠合同？'
            example2_a = '价格：396万元，时间：2013年5月28日'
            example3 = '经审理查明,综艺节目《天天向上》的著作权人为湖南广播电视台,原告经授权,获得了2014年度的该节目在中国大陆地区的独家信息网 络传播权,授权期限自2014年1月1日至2015年12月31日。原告工作人员曾于2014年1月3日向后缀为“pplive.com”和“pptv.com”的邮箱发送权利预警函,告知原告享有涉案节目的独家信息网络传播权。被告确认其中两个收件邮箱地址显示的姓名曾为被告员工,但现已离职。2014年7月25日,国家版权局公布的“2014年度第二批重点影视作品预警名单”中包括涉案节目。根据原告于2015年5月25日办理的证据保全公证,在被告经营的网站(www.pptv.com)首页搜索栏可搜索到涉案节目《天 天向上》不同期数的视频,点击均可正常播放,其中包括“《天天向上》20140620”。审理中,原告确认被告网站上已删除涉案视频。原告主张为本案支出律师费3,000元,但未提交相应证据。以上事实,由原、被告的当庭陈述,原告提供的刻录光盘、作品登记证书、“湖南省人民政府关于组建湖南广播电视台的通知”、相应授 权书及说明书、确认函、国家版权局网页打印件、(2015)巨证经字第122号公证书、(2016)沪普证经字第118号公证书等经庭审质证的证据予以证实,本院予以确 认。被告提交了其后台打印记录,以证明其于2015年3月19日上传涉案视频,并于同年10月22日删除。原告以该证据系被告单方制作为由不予认可。本院对原告的 上述质证意见予以采纳。原告获授权期限及其主张为本案支出律师费用金额？'
            example3_a = '期限：自2014年1月1日至2015年12月31日，金额：3,000元'
            example4 = '当事人围绕诉讼请求依法提交了证据,本院组织当事人进行了证据交换和质证。彭x0和李1向本院提交了死亡证明、房产证、公有住房买卖协议书、协议书及附加协议书。彭x4向本院提交了光明日报房产调查函。原、被告各方对上述证据的真实性均没有异议,本院予以确认并在卷佐证。本院认定 如下事实:2002年4月10日彭x3与李1登记结婚,双方均为再婚,婚后无子女。彭x3与前妻育有一女彭x4、一子彭x0。李1与前夫的育有一子朱x、一女朱x1。彭x3与 李1再婚时,彭x0、彭x4、朱x、朱x1均已成年。2003年1月彭x3与中国科学技术馆签订了公有住房买卖协议书,购买了北京市西城区裕中东里x号楼x门xxx号房屋。2016年1月20日,彭x3去世。2016年1月22日,李1、彭x0、彭x4及朱x、朱x1签订了协议书,约定诉争房屋待李1不在此房屋居住时,将该房产变卖,不得出租或由他人居住,房产变卖时,需四个儿女到场,变卖款的百分之五十由其爱人李1继承,剩余的百分之五十由其儿子彭x0继承。同日,彭x0与彭x4签订附加协议,约定现彭x0愿 将全部遗产所得的百分之三十五交给彭x4,归彭x4所有,全部遗产所得的百分之六十五由彭x0所得。本院向在京中央和国家机关住房交易办公室调查诉争房屋是否可以上市交易,得到复函:诉争房屋可在完善职工住房档案情况下,先办理继承过户业务。之后方可按照交易办的审核要求办理上市交易业务。经彭x0申请,本院摇号确定北京百成首信房地产评估有限公司对北京市西城区裕中东里x号楼x-xxx号房屋进行评估,评估单价为108030元/平方米、总价657.9万元。评估费用15100元,已由彭x0预交。彭x3签订住房买卖协议时间及其儿子继承变卖款金额？'
            example4_a = '时间：2003年1月，金额：百分之五十'
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
        max_input_tokens = 300
        for i, messages in enumerate(bqs):
            new_batch_input = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            max_input_tokens = max(max_input_tokens, len(new_batch_input))
            batch_inputs.append(new_batch_input)
        gen_kwargs = {
            "max_input_tokens": max_input_tokens,
            "max_new_tokens": 200,
            "do_sample": True,
            "top_p": 0.8,
            "temperature": 0.8,
            "num_beams": 1,
        }

        batch_responses = batch(model, tokenizer, batch_inputs, **gen_kwargs)
        for response,label in zip(batch_responses,bls):
            count+=1
            tem = 0
            num+=len(label)
            for i in label:
                if i in response:
                    tem += 1
                    num1 +=1
            if tem == len(label):
                right += 1
            if count % 100 == 0:
                print(label, "======", response)
                print(count, right, float(right / count),num1,num,float(num1/num))
    print(count, right, float(right / count),num1,num,float(num1/num))