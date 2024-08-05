import jieba
from rouge_chinese import Rouge
rouge = Rouge()
batch_responses=['该法典体现了“个人最大限度的自由，法律最小限度的干涉”这一立法精神','该法典具有鲜明的革命性和时代性','该法典的影响后来传播到美洲、非洲和亚洲广大地区']
bls=['该法典体现了“个人最大限度的自由，法律最小限度的干涉”这一立法精神','该法典首次全面规定了法人制度','该法典具有鲜明的革命性和时代性']
res = [' '.join(jieba.cut(re)) for re in batch_responses]
las = [' '.join(jieba.cut(la)) for la in bls]
scores=rouge.get_scores(res, las,avg=True)
print(scores)
