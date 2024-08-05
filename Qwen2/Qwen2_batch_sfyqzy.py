import sys,json,jieba
from rouge_chinese import Rouge
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = sys.argv[1] #
device = "cuda" # the device to load the model onto
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

filename = '../LLMData/按任务打包/涉法舆情摘要/train0.jsonl'
rouge=Rouge()
count = rouge1_f = rouge2_f = rougel_f = t_len = s_len = 0
with (open(filename, 'r', encoding='utf-8') as file):
    for line in file:
        json_line = json.loads(line.strip())
        count+=1
        statement = json_line['text'][:1500]
        summary = json_line['summary']
        instruct = "请返回给定文本相应的一百字左右的摘要。"
        s = statement
        messages = [
            {"role": "system", "content": instruct},
            {"role": "user", "content": s}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(device)
        generated_ids = model.generate(**model_inputs, max_new_tokens=150)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in
                         zip(model_inputs.input_ids, generated_ids)]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        scores = rouge.get_scores(' '.join(jieba.cut(response)), ' '.join(jieba.cut(summary)))
        # print(scores)
        rouge1_f += scores[0]['rouge-1']['f']
        rouge2_f += scores[0]['rouge-2']['f']
        rougel_f += scores[0]['rouge-l']['f']
        if count % 100 == 0:
            print(count, summary, '=============', response)
            print(count, float(rouge1_f/count), float(rouge2_f/count), float(rougel_f/count), 0.2*float(rouge1_f/count)+0.4*float(rouge2_f/count)+0.4*float(rougel_f/count))

print(count, float(rouge1_f/count), float(rouge2_f/count), float(rougel_f/count))