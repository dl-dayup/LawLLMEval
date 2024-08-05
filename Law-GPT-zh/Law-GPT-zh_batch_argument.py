import torch,json
from transformers import AutoTokenizer, AutoModel
from model.modeling_chatglm import ChatGLMForConditionalGeneration

tokenizer = AutoTokenizer.from_pretrained("model", trust_remote_code=True)
model_class = ChatGLMForConditionalGeneration
model = model_class.from_pretrained("model", device_map = "auto").half().eval()

filename = '../../LLMData/按任务打包/论辩理解/stage_1/train_entry0.jsonl'
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
        dlen+=len(s)
        response = model.chat(tokenizer, s[:4096], history='', max_length=4096)
        if str(label) in response:
            right+=1
        if '1' in response and '2' in response and '答案：' + str(label) not in response:
            right -= 1
        if count % 100 == 0:
            print(count, label, '=============', response)
            print(count, right, float(right / count))

        # if 'A' in response and 'B' in response and 'C' in response and 'D' in response and '答案：' + answer not in response:
        #     right -= 1
print(count,right,float(right/count),float(dlen/count))