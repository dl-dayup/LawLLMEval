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
    filename = '../../LLMData/按任务打包/可解释类案匹配/mytrain0.json'
    label_dict = {0: "完全不相关", 1: "基本不相关", 2: "比较相关", 3: "非常相关"}
    all_labels = label_dict.values()
    count = num = right = dlen = 0
    with (open(filename, 'r', encoding='utf-8') as file):
        for line in file:
            json_line = json.loads(line.strip())
            label = json_line['label']
            text_a = json_line['Case_A']
            textA = "".join(a for a in text_a)
            text_b = json_line['Case_B']
            textB = "".join(b for b in text_b)
            instruct = "根据A、B两段话的匹配程度，判断属于以下标签中的哪一个？[非常相关、比较相关、基本不相关、完全不相关]。非常相关：要件事实相关，且案情事实相关。比较相关：要件事实相关，但案情事实不相关。基本不相关：要件事实不相关，但案情事实相关。完全不相关：要件事实不相关，且案情事实不相关。"
            example1 = 'A：河南省郑州市金水区人民检察院指控:2019年5月, 被告人XXX为谋取利益,在明知个人信息可能被犯罪分子利用的情况下,使用个人信息在郑州市金水区注册郑州莉青网络科技有限公司、郑州恰勇文化传媒有限公 司,在郑州市金水区的光大银行丰产路支行、工商银行军区支行办理对公账户,并将以自己身份证注册的、郑 州恰勇传媒有限公司的手续及上述两公司的对公账户银行卡、U盾、结算卡、身份证复印件、捆绑对公账户的手机卡等相关资料,分别以5500元、4300元的价格卖给XXX。 经查询,XXX卖出的郑州莉青网络科技有 限公司对公账户被诈骗分子利用,支付结算金额70万元左右。 被告人XXX到案后自愿如实供述自己的罪行。 针 对上述指控的事实,公诉机关向本院提供了 被告人XXX的供述,公司注册登记材料,对公账户信息,银行交易明细,公安部反诈平台查询结果,广昌县***立案登记情 况,受案、到案经过,户籍证明等证据。  据此认为被告人XXX的行为已触犯了《中华人民共和国刑法》第二百八十七条之二第一款之规定,应当以帮助信息网络犯罪活动罪追究其刑事责任。 建议对被告人XXX判处有期徒刑六个月,并处罚金。 提请本院依法惩处。 经法庭审理,查明的事实、证据与指控的事实、证据相同。 被告人XXX对起诉书指控的事实、证据及量刑建议均不持异议,且被告人XXX已签字具结。 本院予以确认。 本院认为:被告人XXX明知他人利用信息网络实施犯罪,为其犯罪提供支付 结算等帮助,情节严重,其行为已构成帮助信息网络犯罪活动罪,公诉机关指控被告人XXX犯帮助信息网络犯罪活动罪,罪名及提出的量刑建议 成立,依法予以支持。 被告人XXX到案后如实供述自己的罪行,自愿认罪认罚,均可从轻处罚。B：经审理查明,被告人XXX在2019年8、9月份,明知他人利用自己的银行卡可能从事违法犯罪活动,仍在多家银行办理6张银行卡(包括中国农业银行卡,XXX)及配套U 盾、手机卡出借给初中同学XXX,共获利18900元。 经***查证,被告人XXX的农业银行卡(卡号62 78)自2019年10月19日至2020年7月19日累计入账金额达9215995元。 认定上述事实的证据有被告人XXX的常住人口基本信息表、前 科情况查询证明、到案经过;方城县***受案登记表、立案决定书、拘留逮捕手续等;***调取证 据通知书、检查笔录、XXX农业银行卡交易明细清单、支付宝转账 记录截图;曾鹏、XXX、XXX、靖春勇等人供述;被告人XXX的供述与辩解;认罪认罚具结书、量刑建议书等证据材料在卷佐证。 以上证据已经法庭当庭示证、质证,证据来源合法,且能相互印证,被告人XXX犯帮助信息网络犯罪活动罪事实清楚,证据确实充分, 足以认定。 本院认为,被告人XXX明知他人利用信息网络实施犯罪活动而提供资金结算的银行卡,情节严重,其行为已构成帮助信息网络犯罪活动罪,依法应予惩处。 公诉机关指控罪名成立,本院予以支持。 被告人XXX在归案后能够如实供述自己的罪行,是坦白,可以从轻处罚。 被告人XXX在庭审时表示认罪,且已在公诉机 关自愿签署认罪认罚具结书,可以依法从宽处理。 公诉机关对其提出判处有期徒刑八个月,并处罚金一万元的量刑建议适当,本院予以采纳。 被告人XXX辩称自己不是明知,与查明事实不符,本院不予采信。 辩护人有关被告人XXX具有从轻处罚情节的意见,本院予以采纳;但关于对XXX适用缓刑的意见,因不符合宣告缓刑条件,本院不予采纳。 根据被告人XXX的犯罪事实、性质、情节及社会危害性,依照《中华人民共和国刑法》第二百八十七条之二第一款、第六十七条第三款、第五十二条、第五十三条、第六十四条、第 六十一条之规定,判决如下:'
            example1_a = '比较相关'
            example2 = 'A：公诉机关指控并经本院审理查明,被告人XXX、XXX 在明知XXX(台湾人)等人利用信息网络实施犯罪的情况下,XXX于2017年11月至2019年6月间,为“阿拉爱上海”黄色论坛网站收取广告位费用XXX,非法获利9万余元;XXX于2017年10月至2019年7月利用其自身设立的发卡网为“阿拉爱上海”黄色论坛网站提供充值等保障性服务, 营运额100余万元,非法获利2万余元。 经鉴定,“ 阿拉爱上海”论坛“北京XXX验证发帖区”版块下99个标记“大兴”地区的帖子,其内容均包含交易价格、联系方式( 部分需要使用论坛社区币购买)等涉及性交易买卖 的信息。 2019年7月11日,2019年7月16日,被告人XXX、XXX被分别传唤到案,到案后均如实供述上述事实。 自 二被告人处查获的涉案电脑、手机均已扣押 。 公诉机关向本院移交了指控的证据并在律师的见证下向被告人XXX、XXX告知了认罪认罚从宽制度诉讼权利,签署 了认罪认罚具结书,并认为被告人XXX、XXX具有如实供述的从轻处罚情节,建议判处被告人XXX有期徒刑六个月至一年,并处罚金;建议判处被告人XXX有期徒刑六个月至一年,并处罚金。 被告人XXX、XXX对指控的事实、罪名及量刑建议没有异议并签字具结,且在开庭审理过程中亦无异议。 被告人XXX的辩护人的辩护意见为:被告人XXX系从犯,初犯,偶犯,认罪 认罚,希望能够从轻处罚。 本院认为,公诉机关指控被告人XXX、XXX犯非法利用信息网络罪的事实清楚,证据确实、充分,指控罪名成立,量刑建议适当,应予采纳。 鉴于被告人XXX、XXX自愿认罪认罚,对其从宽处罚。 XXX的辩护人关于其系从犯的辩护意见,本院不予采纳;其余辩护意见, 予以采纳。B：经审理查明,2015年至今,被告人XXX高和XXX共谋在成都市温江区,先后建立四川逍遥网、四川耍耍网、成都耍耍网三个非法网站,在网站上为他人发布卖淫XXX广告。 XXX高主要负责购买域名以及网站的日常运营、管理;XXX负责编写四川逍遥网代码、参与四川逍遥网网站前期经营,并为XXX远高建立、运营四川耍耍网 、 成都耍耍网提供技术支持。 XXX高、XXX通过在前述网站发布卖淫XXX广告,收取信息费用30100元,XXX向XXX转款6400元。 经远程勘验,成都耍耍网( )主页显示总帖数137139、会员总数103842。 另查明,自2016年下半年以来,曾某(已判决)在成都市成华区“某某某小区”暂住地内,通过四川逍遥网、四川耍耍网 等网站发布XXX信息广告,并多次通过微信、XXX介绍卖淫。 2018年7月,成都市***成华区分局治安大队在办理曾某涉嫌介绍卖淫一案时,根据曾某交代线索展 开侦查,于2019 年2月28日在成都市温江区内将XXX高、XXX挡获,自杨远高处查获并扣押其使用的黑色小米手机一部,在二被告人位于成都市温江区的住所查获并扣押运营网站使 用的黑色acer牌笔记本电脑一台、XXX使用的黑色苹果牌XSMAX手机一部。 此外,***还查获并扣押玫瑰金苹果7Plus手机、玫瑰金OPPOR9手机、 红色无品牌手机 各一部以及电脑硬盘三个。 2019年7月23日,被告人XXX高、XXX家属各代为退赃15000元。 认定上述事实有:立案登记表、受案登记表、到案经过,证人曾某的证 言,被告人XXX高、XXX的户籍证明及辨认说明,被告人XXX高、XXX的供述,远程勘验笔录,提取笔录及照片,检查笔录、搜查笔录、扣押笔录、扣押决定书、扣押物 品清单,电子证物检查笔录,退缴票据,认罪认罚具结书等证据予以证实。 本院认为,被告人XXX高、XXX为牟取非法利益,设立网站为他人 实施介绍卖淫的犯罪活 动发布XXX信息,情节严重,其行为符合《中华人民共和国刑法》第二百八十七条之一的规定,构成非法利用信息网络罪,成都市成华区人民 检察院起诉指控被告人所犯罪行事实清楚,罪名正确,本院予以支持。 公诉机关指控被告人XXX高、XXX非法获利数十万,仅有被告人XXX高的供述,本院不予支 持,但XXX高指认其通过微 信收取信息发布费用30100元,本院确认该金额为被告人获利数额,对相应辩护意见予以采纳。 本案系共同犯罪,被告人XXX与XXX共谋通过设立网站为他人介绍卖 淫发布信息以谋取非法利益,被告人XXX高购买域名、负责网站日常运营、管理,被告人XXX提供网站建设、维护的技术支持,二被告人 不宜区分主从,本院在量刑 时根据二被告人的具体犯罪作用予以量刑。 对被告人XXX的辩护人所提XXX系从犯的辩护意见,本院不予采纳。 被告人XXX高、XXX到案后如实供述犯罪事实、认 罪认罚,依法可以从轻处罚。 被告人XXX高、XXX无犯罪前科,且已退缴大部分违法所得,可酌定从轻处罚。 关于辩护人建 议适用缓刑的辩护意见,本院认为,被告人XXX高、XXX建立运营数个网站为他人违法犯罪活动发布信息,严重扰乱社会公共管理秩序,不宜适用缓刑,本院对相应辩 护意见不予采纳。 扣押在案的黑色苹 果牌XSMAX手机、黑色小米手机、黑色acer牌笔记本电脑系被告人实施犯罪所用作案工具,应予没收,其他物品由扣押机 关依法处置。 综上,为了维护公共网络通讯的正常秩序,保障公众的合法权益,惩罚犯罪,经本院审判委员会讨论决定,依照《中华人民共和国刑法》第二百八 十七条之一、第二十五条、第六十七条第三 款、第五十二条、第五十三条、第六十四条之规定,判决如下:'
            example2_a = '基本不相关'
            example3 = 'A：经审理查明,2017年2月11日,被告人XXX通过网络认识XXX(已判处刑罚)并从其手中以每张400元价格两次购买三张他人居民身份证。 被告人XXX到案后如实供述 自己的罪行。 上述事实,被告人XXX在开庭审>理过程中无异议,且有其在侦查阶段的供述、证人XXX的证言、刑事判决书、武陟县***2018年9月25日证明、户籍证明、无犯罪记录证明、发破案经过、抓获证明、武陟县***2019年6月21日证明和2019年12月21日证明等证据证实,足以认定。 本院认为,被告人XXX买卖居民身份证,其行为构成买卖居民身份证件罪。 公诉机关对被告人XXX买卖身份证件罪的指控成立。 被告人XXX主动投案,到案后如实供述自己罪行,系自首,且愿意接受处罚,对其可从轻处罚。 根据被告人XXX的犯罪情节及悔罪表现,适用缓刑对所居住社区没有重大不良影响,可宣告缓刑。 公诉机关的量刑建议适当,予以采纳。B：渑池县人民检察院指控:2018年6月底,被告人XXX在渑池县东方希望集团南门捡到XXX驾驶证,把XXX照片去掉,把自己照片贴在XXX驾驶证上,后XXX使用伪造的驾驶证驾驶车牌号为>豫M (豫MD987挂)的半挂车,应对交警检查,并使用变造的驾驶证处理七次违章罚款,直至被渑池县***交警大队民警查获。 针对上述指控犯罪事 实,公诉机关提供了查获经过、扣押清单及照片、破案报告、违章记录等书证,被告人XXX的供述等证据。 指控认为,被告人XXX变造驾驶证一本,并使用变造的驾 驶证,其行为应当以变造身份证件罪追究刑事责任。 被告人XXX认罪认罚,建议判处拘役三个月,缓刑四个月,并处罚金3000元。 本院认为,公诉机关指控被告人XXX犯变造身份证件罪罪名成立,依法予以支持。 被告人XXX有坦白情节,认罪认罚且签字具结,公诉机关提出的量刑建议适当,本院予以支持。 依照《中华人民共> 和国刑法》第一百三十三条之一第一款、第五十二条、第五十三条、第六十七条第三款、第七十二条第一款、第三款、第七十三条第一款、第三款之规定,判决如下:'
            example3_a = '完全不相关'
            example4 = 'A：经审理查明,2017年1月至2018年5月间,被告人XXX>伙同安全员XXX(已判决)、XXX东(另案处理)等人,以失误预警、帮踩刹车等违反考试规定的方式,组织446名驾校学员在机动车驾驶人考试科目三中实施作弊,被告人XXX收取每名学员、人民币500至600元不等的好处费后,向XXX支付考试作弊好处费共计人民币58400元,向XXX东支付考试作弊好处费共计人民币120000元,从中>非法获利共计人民币50000余元。 案发后,被告人XXX已退缴违法所得人民币50000元。 被告人XXX在侦查及审查起诉、审理阶段均自愿认罪。 上述事>实,被告人XXX在开庭审理过程中亦无异议,且有扣押笔录,扣押决定书,扣押清单,业务凭证,执法暂扣款票据,微信转账记录,情况说明,户籍证明,抓获经过,被告人XXX的供述和辩解,同案犯XXX、XXX东的供述和辩解等证据在案证实,足以认定。 本院认为,被告人XXX在法律规定的国家考试中,组织作弊,其行为已构成组织>考试作弊罪,依法应予以惩处。 公诉机关指控的事实和罪名成立,适用法律正确。 被告人XXX在侦查阶段、审查起诉阶段及本院审理阶段均自愿认罪,且已退缴违法所得,本院酌情予以从轻处罚。 根据被告人XXX的犯罪事实、性质、对社会的危害程度及认罪表现,依法确定其刑罚。B：经审理查明,2018年11月至12月期间,被告人XXX为谋取非法利益,通过QQ联系向被告人XXX以1500元的价格购买了2019年全国硕士研究生招生考试政治、英语及MBA联考答案及部>分试题,后在位于本市万柏林区“十二院城”小区1号楼2单元3205的家中,通过QQ以每一科1500元至2500元的价格出售给考生XXX、XXX、XXX、XXX、XXX、XXX、高某1等人。 后被告人XXX又通过QQ联系被告人XXX以2000元左右的价格向考生XXX、XXX、XXX、高某1出售无线接收器、发射器等作弊器材。 被告人XXX收取作弊器材定金后,将考生收货地址发给被告人XXX,被告人XXX安排被告人XXX将作弊器材发货给考生并收取剩余货款。 经鉴定,被告人XXX向被告人XXX购买 并出售给考生的试题及答案内容与2019全国硕士研究生招生考试试题内容、关键词一致。 证实上述事实的证据有:金仙”了。 也没有接收到答案。 XXX扣押>笔 记本电脑1台、手机1部、广告贴字1包。 以上证据,均经当庭举证、质证,且证据间能相互印证,足以认定。 本院认为,被告人XXX在法律规定的国家考试中,组织 考生作弊,被告人XXX、XXX为他人实施考试作弊提供作弊器材,三被告人的行为已构成组织考试作弊罪;被告人XXX为他人实施考试作弊向他人非法出售考>试试题 及答案,其行为已构成非法出售试题、答案罪。 公诉机关指控的罪名成立。 关于被告人XXX的辩护人提出犯罪中止的辩护意见。 经查,被告人XXX是在***查处犯罪过程中被迫中止,不具有中止犯罪的主动性,该辩护意见不成立,本院不予采纳。 关于被告人XXX、XXX的辩护人提出二被告人系从犯的辩护意见。 经查,二被告人在共同犯罪中,分工合作,具体实施了犯罪行为,不宜认定为从犯,故对该辩护意见,本院不予采纳。 被告人XXX、XXX、XXX、XXX当庭自愿认罪,如实供述犯罪事 实,可以从轻处罚。 对被告人及辩护人的其他合理辩护意见,本院予以采纳。 依照《中华人民共和国刑法》第二百八十四条之>一第一款、第二款、第六十一条 、第五十二条、第五十三条、第六十七条第三款、第六十四条的规定,判决如下:'
            example4_a = '比较相关'
            s = "A：" + textA + "。B：" + textB
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
            messages.append({"role": "user", "content": s[:3000]})
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
            if label_dict[label] in response:
                right += 1
            if '完全不相关' in response and '基本不相关' in response and '比较相关' in response and '非常相关' in response and '答案：' + \
                    label_dict[label] not in response:
                right -= 1
            if count % 100 == 0:
                print(label_dict[label], "======", response)
                print(count, right, float(right / count))
    print(count, right, float(right / count))