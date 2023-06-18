import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel

# 加载预训练模型和分词器
model_name = "../../chatglm-6b/"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, load_in_8bit=True, trust_remote_code=True, device_map='auto')

# # 读取lora模型参数
model_path = "./output"
model = PeftModel.from_pretrained(model, model_path, torch_dtype=torch.long)
model.eval()

# 输入对话历史和需要输出的长度
news_text = "content：买主花86万买到“凶宅”索赔，法院：构成欺诈，未告知其前妻自杀，全款返还,summary: "
input_ids = tokenizer.encode(news_text, return_tensors='pt').long()
out = model.generate(
        input_ids=input_ids,
        max_length=150,
        temperature=0
    )
answer = tokenizer.decode(out[0])
print('新闻：', answer)

