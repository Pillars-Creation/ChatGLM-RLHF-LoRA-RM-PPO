import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel

# 加载预训练模型和分词器
model_name = "../../chatglm-6b/"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, load_in_8bit=True, trust_remote_code=True, device_map='auto')

# # 读取lora模型参数
model_path = "./path_to_rm_checkpoint/checkpoint-300"
model = PeftModel.from_pretrained(model, model_path, torch_dtype=torch.long)
model.eval()

# 输入对话历史和需要输出的长度
text = "在['俄驻美大使馆：美制裁证实了俄罗斯追求金融和技术主权的准确性 (新闻)', 'EDG战胜OMG后，五队确定冒泡赛资格 ！杰杰点评自己像明凯 (游戏)', '偷瓜贼“小刺猬”把脑袋插西瓜里偷吃，当事人：它没有那么大的破坏力，吃就吃点吧 (生活)', '中俄多领域对话合作齐头并进 (军事)', '话筒NBA｜库里：生涯初期詹姆斯告诉我如何度过困难 他是我的朋友 (体育)', '和我一起来玩木头吧！ (生活)', '年幼的雄性海狗练习搏斗，它们必须在成年时掌握这项技能 (纪录片)',  "
quary = "从上面文章中挑选出体育相关的新闻"
input = text + quary
input_ids = tokenizer.encode(input, return_tensors='pt').long()
out = model.generate(
        input_ids=input_ids,
        max_length=2048,
        temperature=0
    )
answer = tokenizer.decode(out[0])
print('新闻：', answer)

