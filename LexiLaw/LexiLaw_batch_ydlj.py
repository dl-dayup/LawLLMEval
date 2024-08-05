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

filename = '../LLMData/按任务打包/阅读理解/ydlj_train.json'
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
        response = generate(model, tokenizer, s[:4090])
        print(response,'=======',answers,"+++++++++"+s)
        if len(response) <len(s)-300:
            tem=0
            for i in answers:
                if i in response:
                    tem += 1
            if tem == len(answers):
                right += 1
            print(count,right,float(right/count))
print(count,right,float(right/count))