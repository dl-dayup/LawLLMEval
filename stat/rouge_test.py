# from rouge import Rouge
from rouge_chinese import Rouge
import jieba
a="四川宏达集团实际控制人刘沧龙收到不利消息，其部分银行账户中的资金被云南省高级人民法院冻结，可能对公司日常生产经营造成一定影响。宏达股份正持续关注该事项的进展，并按法律法规及时履行相应的信息披露义务。"
b="四川宏达集团部分资金被冻结，宏达股份表示尚未收到相关法律文件和通知。"
rc=Rouge()
scores=rc.get_scores(' '.join(jieba.cut(a)),' '.join(jieba.cut(b)))
print(scores)