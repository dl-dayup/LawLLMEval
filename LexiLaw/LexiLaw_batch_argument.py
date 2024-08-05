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

filename = '../LLMData/按任务打包/论辩理解/stage_1/train_entry0.jsonl'
count=num=right=dlen=0
with open(filename, 'r', encoding='utf-8') as file:
    for line in file:
        json_line = json.loads(line.strip())
        count += 1
        sc = json_line['sc']
        crime = json_line['crime']
        bc_1 = json_line['bc_1']
        bc_2 = json_line['bc_2']
        bc_3 = json_line['bc_3']
        bc_4 = json_line['bc_4']
        bc_5 = json_line['bc_5']
        label = json_line['answer']
        s = sc + "在这个关于"+crime+"的描述中，以下五个句子中有一个句子可以与给定文本能组成有争议的观点对，请判断是哪一个句子并返回句子号。不需要重复原文内容和给定的五个句子内容。1："+bc_1+"。2："+bc_2+"。3："+bc_3+"。4: "+bc_4+"。5: "+bc_5
        # dlen+=len(s)
        response = generate(model, tokenizer, s[:4090])
        # print(count, label, '======',response)
        if "句子号是"+str(label) in response[:30]:
            right+=1
        # if 'A' in response and 'B' in response and 'C' in response and 'D' in response and '答案：' + answer not in response:
        #     right -= 1
        print(count, right, float(right / count))
print(count,right,float(right/count),float(dlen/count))