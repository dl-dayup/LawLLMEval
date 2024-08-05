import torch,json
from transformers import AutoModel
from modeling_chatglm import ChatGLMForConditionalGeneration
from tokenization_chatglm import ChatGLMTokenizer
from peft import PeftModel

def generate(model,tokenizer,text):
    with torch.no_grad():
        input_text = text
        ids = tokenizer.encode(input_text)
        input_ids = torch.LongTensor([ids]).cuda()
        output = model.generate(
            input_ids=input_ids,
            min_length=20,
            max_length=4096,
            do_sample=False,
            temperature=0.7,
            num_return_sequences=1
        )[0]
        output = tokenizer.decode(output)
    return output.strip()

model_path = './model/'
model = ChatGLMForConditionalGeneration.from_pretrained(model_path, trust_remote_code=True).eval()
tokenizer = ChatGLMTokenizer.from_pretrained(model_path, trust_remote_code=True)
torch.set_default_tensor_type(torch.cuda.FloatTensor)
model.half().cuda()

filename = '../LLMData/按任务打包/司法考试/0_train0.json'
count=num=right=dlen=0
with open(filename, 'r', encoding='utf-8') as file:
    for line in file:
        # 去除每行末尾的换行符并解析JSON
        json_line = json.loads(line.strip())

        answers = json_line['answer']
        if len(answers) > 1:
            count += 1
            answer = answers[0]
            option_list = json_line['option_list']
            statement = json_line['statement']
            example = '参考如下例子回答单项选择题。例子1：以下有关民事简易程序的说法中，正确的有?A：简易程序的审限为三个月，属于法定不可变期间。B：法院在审理案件过程中发现案情复杂，需要转为普通程序的，应当在审限届满前作出决定并通知当事人。C：简易程序的开庭方式较为灵活便捷，经过双方当事人同意，法院可以采用同步视频等视听传输技术的方式开庭。D：适用简易程序案件的举证期限可以由法院确定，也可由当事人协商并经法院准许，但不得超过30日.答案：C。'
            example2 = '例子2：下列关于自然人民事权利能力和民事行为能力法律适用的说法，符合我国《涉外民事关系法律适用法》规定的是:A：自然人的民事权利能力，适用经 常居所地法律，但涉及婚姻家庭、继承的除外。B：自然人从事民事活动，依照经常居所地法律为无民事行为能力，依照行为地法律为有民事行为能力的，一律适用行为地法律。C：自然人的民事权利能力，一律适用经常居所地法律。D：自然人的民事行为能力，一律适用经常居所地法律。答案：C。'
            instruct = "请回答如下多项选择题，有2-4个选项为正确选项，需返回符合题目的所有选项。"
            s = instruct + statement + "。A：" + option_list['A'] + "。B：" + option_list['B'] + "。C：" + option_list[
             'C'] + "。D：" + option_list['D']
            dlen=len(s)
            # print('statement', s)
            response = generate(model, tokenizer, s[:4090])
            print(answer, "======", response)
            tem = 0
            for answer in answers:
                if answer in response:
                    tem += 1
            if 'A' in response and 'B' in response and 'C' in response and 'D' in response:
                tem = 4
            if tem == len(answers):
                right += 1
print(count, right, float(right / count))