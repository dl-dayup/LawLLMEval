from transformers import AutoTokenizer, AutoModel
from model.modeling_chatglm import ChatGLMForConditionalGeneration
import sys,os,math,json,torch

tokenizer = AutoTokenizer.from_pretrained("model", trust_remote_code=True)
model_class = ChatGLMForConditionalGeneration
model = model_class.from_pretrained("model", device_map = "auto").half().eval()
s="1804年的《法国民法典》是世界近代法制史上的第一部民法典，是大陆法系的核心和基础。下列关于《法国民法典》的哪一项表述不正确?A:该法典体现了“个人最大限度的自由，法律最小限度的干涉”这一立法精神, B:该法典具有鲜明的革命性和时代性, C:该法典的影响后来传播到美洲、非洲和亚洲广大地区, D:该法典首次全面规定了法人制度"
response = model.chat(tokenizer, s, history='', max_length=500)
print(response)