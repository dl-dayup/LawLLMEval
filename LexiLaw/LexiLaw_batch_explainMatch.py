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

filename = '../LLMData/按任务打包/可解释类案匹配/mytrain0.json' #competition_stage_2_train0.json'
label_dict= {0:"完全不相关", 1:"基本不相关", 2:"比较相关", 3:"非常相关"}
all_labels=label_dict.values()
count=num=right=dlen=0
with open(filename, 'r', encoding='utf-8') as file:
    for line in file:
        json_line = json.loads(line.strip())
        count += 1
        label = json_line['label']
        textA = json_line['Case_A']
        textB = json_line['Case_B']
        # relation = json_line['relation']
        # matchs=[]
        # for r in relation:
        #     print((text_a[r[0]],text_b[r[1]]))
        #     matchs.append((text_a[r[0]],text_b[r[1]]))

        s = "根据A、B两段话的匹配程度，进行分类任务，判断属于以下标签中的哪一个？[非常相关、比较相关、基本不相关、完全不相关]。A：" + textA + "。B：" + textB
        dlen+=len(s)
        # print('statement', s)
        response = generate(model, tokenizer, s[:4090])# print(answer, response)
        print(count, label, "======", response)
        if label_dict[label] in response:
            right += 1
        if '完全不相关' in response and '基本不相关' in response and '比较相关' in response and '非常相关' in response and '答案：' + label_dict[label] not in response:
            right -= 1
        print(count,right,float(right/count))
print(count,right,float(right/count),float(dlen/count))