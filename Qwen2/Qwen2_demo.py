import sys
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = sys.argv[1] #
print(model_name)
device = "cuda" # the device to load the model onto
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)
instruct="请回答如下单项选择题，只需返回一个正确选项。"
s="1804年的《法国民法典》是世界近代法制史上的第一部民法典，是大陆法系的核心和基础。下列关于《法国民法典》的哪一项表述不正确?A:该法典体现了“个人最大限度的自由，法律最小限度的干涉”这一立法精神, B:该法典具有鲜明的革命性和时代性, C:该法典的影响后来传播到美洲、非洲和亚洲广大地区, D:该法典首次全面规定了法人制度"
messages = [
    {"role": "system", "content": instruct},
    {"role": "user", "content": s}]
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
model_inputs = tokenizer([text], return_tensors="pt").to(device)
generated_ids = model.generate( **model_inputs, max_new_tokens=512)
generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
# print(response[response.find('assistant'):])

