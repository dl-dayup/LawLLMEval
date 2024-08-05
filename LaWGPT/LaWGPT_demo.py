import sys,json,fire,torch
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from utils.prompter import Prompter
class Infer():
    def __init__(
        self,
        load_8bit: bool = False,
        base_model: str = "",
        lora_weights: str = ""
    ):
        tokenizer = LlamaTokenizer.from_pretrained(base_model)
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=load_8bit,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        try:
            print(f"Using lora {lora_weights}")
            model = PeftModel.from_pretrained(
                model,
                lora_weights,
                torch_dtype=torch.float16,
            )
        except:
            print("*"*50, "\n Attention! No Lora Weights \n", "*"*50)
            
        # unwind broken decapoda-research config
        model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
        model.config.bos_token_id = 1
        model.config.eos_token_id = 2
        if not load_8bit:
            model.half()  # seems to fix bugs for some users.
        model.eval()

        if torch.__version__ >= "2" and sys.platform != "win32":
            model = torch.compile(model)
        self.base_model = base_model
        self.lora_weights = lora_weights
        self.model = model
        self.tokenizer = tokenizer
    def generate_output(
        self,
        instruction,
        input=None,
        temperature=0.1,
        top_p=0.75,
        top_k=40,
        num_beams=1,
        max_new_tokens=256,
        **kwargs,
    ):
        # prompt = self.prompter.generate_prompt(instruction, input)
        inputs = self.tokenizer(instruction+input, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.model.device)
        generation_config = GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            # repetition_penalty=10.0,
            **kwargs,
        )
        with torch.no_grad():
            generation_output = self.model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
            )
        s = generation_output.sequences[0]
        output = self.tokenizer.decode(s)
        return output #self.prompter.get_response(output)

    def infer_from_file(self, infer_data_path):
        with open(infer_data_path) as f:
            for line in f:
                data = json.loads(line)
                instruction = data["instruction"]
                output = data["output"]
                print('=' * 100)
                print(f"Base Model: {self.base_model}    Lora Weights: {self.lora_weights}")
                print("Instruction:\n", instruction)
                model_output = self.generate_output(instruction)
                print("Model Output:\n", model_output)
                print("Ground Truth:\n", output)
                print('=' * 100)


load_8bit: bool = False
base_model: str = "./models/base_models/legal_base-7b/"
lora_weights: str = "./models/lora_weights/legal-lora-7b/"
prompt_template: str = "" # The prompt template to use, will default to alpaca.
infer_data_path: str = ""

infer = Infer(
    load_8bit=load_8bit,
    base_model=base_model,
    lora_weights=lora_weights
)

try:
    infer.infer_from_file(infer_data_path)
except Exception as e:
    print(e, "Read infer_data_path Failed! Now Interactive Mode: ")
    # text = "1804年的《法国民法典》是世界近代法制史上的第一部民法典，是大陆法系的核心和基础。下列关于《法国民法典》的哪一项表述不正确?A: 该法典体现了“个人最大限度的自由，法律最小限度的干涉”这一立法精神, B: 该法典具有鲜明的革命性和时代性, C: 该法典的影响后来传播到美洲、非洲和亚洲广大地区, D: 该法典首次全面规定了法人制度"
    # instruction = "你是中国顶尖智能法律顾问 LaWGPT，具备强大的中文法律基础语义理解能力，能够出色地理解和执行与法律问题和指令。你只能回答与中国法律领域相关的问题，其余领域的问题请礼貌地拒绝回答。接下来，请依据中国法律来回答下面这个问题。请回答如下单项选择题，只需返回一个正确选项。"
    # instruction = "你是中国顶尖智能法律顾问 LaWGPT，具备强大的中文法律基础语义理解能力，能够出色地理解和执行与法律问题和指令。你只能回答与中国法律领域相关的问题，其余领域的问题请礼貌地拒绝回答。接下来，请依据中国法律来回答下面这个问题。请列出上述句子中涉及到的人名、地名、时间、毒品类型、毒品重量，同时判断这些人物和毒品的关系是贩卖（给人），贩卖（毒品），持有，非法容留中的哪一种，给出关系的时候需要指明头尾实体是具体的人物或毒品类型"
    # text = "经审理查明，2015年7月22日1时许，公安民警接到群众吴某某举报称贵阳市云岩区纯味园宿舍有一男子持有大量毒品。公安民警接警后前往举报地点搜查。在搜查过程中，从被告人焦某某身上查获毒品一包，经刑事科学技术鉴定检出海洛因计重120克。涉案毒品已上交省公安厅禁毒总队。"
    instruction = "你是中国顶尖智能法律顾问 LaWGPT，具备强大的中文法律基础语义理解能力，能够出色地理解和执行与法律问题和指令。你只能回答与中国法律领域相关的问题，其余领域的问题请礼貌地拒绝回答。接下来，请依据中国法律来回答下面这个问题。问题：酒驾撞人怎么判刑？"

    print(infer.generate_output(instruction,""))