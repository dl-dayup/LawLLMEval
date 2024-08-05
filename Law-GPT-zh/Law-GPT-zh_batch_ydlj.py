import torch,json
from transformers import AutoTokenizer, AutoModel
from model.modeling_chatglm import ChatGLMForConditionalGeneration

tokenizer = AutoTokenizer.from_pretrained("model", trust_remote_code=True)
model_class = ChatGLMForConditionalGeneration
model = model_class.from_pretrained("model", device_map = "auto").half().eval()

filename = '../../LLMData/按任务打包/阅读理解/ydlj_train.json'
count=num=right=dlen=0
with open(filename, 'r', encoding='utf-8') as file:
    data = json.load(file)['data']
    for d in data:
        count += 1
        p = d['paragraphs'][0]
        context = p['context']
        casename = p['casename']
        qas = p['qas'][0]
        question = qas['question']
        answers_q = qas['answers'][0]
        answers=[]
        for i in answers_q:
            answers.append(i['text'])
        example = ''
        s = example + context + '请问在这个'+casename+'中，'+question+'，不需要重复原文内容，只返回答案部分即可，如果文中没有提及请返回无法回答'
        # print('statement', s)
        response = model.chat(tokenizer, s[:4096], history='', max_length=4096)
        print(count, response)

        if len(response) < len(s) - 300:
            tem=0
            for i in answers:
                if i in response:
                    tem += 1
            if tem == len(answers):
                right += 1
            print(count,right,float(right/count))
print(count,right,float(right/count))